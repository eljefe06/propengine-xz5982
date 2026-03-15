<?php
namespace MyRock\MailEngine\Services;

defined( 'ABSPATH' ) || exit;

use MyRock\MailEngine\Models\Automation;
use MyRock\MailEngine\Models\Contact;
use MyRock\MailEngine\Mail\MailManager;

class AutomationService {

	/**
	 * Trigger all active automations matching a given trigger type for a contact.
	 *
	 * For each automation that matches, the first step is enqueued with the
	 * appropriate run_at time based on the step's delay settings.
	 *
	 * @param int    $contact_id
	 * @param string $trigger  e.g. 'subscribe', 'unsubscribe', 'tag_added'.
	 * @return void
	 */
	public static function trigger_for_contact( int $contact_id, string $trigger ): void {
		$automations = Automation::get_active_by_trigger( $trigger );

		if ( empty( $automations ) ) {
			return;
		}

		foreach ( $automations as $automation ) {
			$automation_id = (int) $automation['id'];
			$steps         = Automation::get_steps( $automation_id );

			if ( empty( $steps ) ) {
				continue;
			}

			// Sort by step_order ascending to get the first step.
			usort( $steps, static fn( $a, $b ) => (int) $a['step_order'] <=> (int) $b['step_order'] );

			$first_step = $steps[0];
			$run_at     = self::calculate_run_at( (int) $first_step['delay_value'], $first_step['delay_unit'] );

			Automation::enqueue( [
				'automation_id' => $automation_id,
				'step_id'       => (int) $first_step['id'],
				'contact_id'    => $contact_id,
				'status'        => 'pending',
				'run_at'        => $run_at,
				'created_at'    => current_time( 'mysql' ),
			] );
		}
	}

	/**
	 * Process all pending automation queue items whose run_at time has arrived.
	 *
	 * Intended to be called by a WP-Cron hook every 5 minutes.
	 *
	 * @return void
	 */
	public static function process_pending_steps(): void {
		$items = Automation::get_pending_queue_items( 20 );

		foreach ( $items as $item ) {
			self::execute_step( $item );
		}
	}

	/**
	 * Execute a single automation queue item.
	 *
	 * Supported step types: email, wait, tag, untag, condition.
	 *
	 * @param array $queue_item  A queue row as returned by Automation::get_pending_queue_items().
	 * @return void
	 */
	public static function execute_step( array $queue_item ): void {
		$queue_item_id = (int) $queue_item['id'];
		$automation_id = (int) $queue_item['automation_id'];
		$step_id       = (int) $queue_item['step_id'];
		$contact_id    = (int) $queue_item['contact_id'];

		// Mark in-progress immediately to prevent duplicate processing.
		Automation::update_queue_item( $queue_item_id, [ 'status' => 'processing' ] );

		$step    = Automation::get_step( $step_id );
		$contact = Contact::find( $contact_id );

		if ( ! $step || ! $contact ) {
			Automation::update_queue_item( $queue_item_id, [
				'status'     => 'failed',
				'error'      => ! $step ? 'Step not found.' : 'Contact not found.',
				'updated_at' => current_time( 'mysql' ),
			] );
			return;
		}

		$step_type    = $step['type'] ?? '';
		$step_data    = isset( $step['data'] ) && is_array( $step['data'] )
			? $step['data']
			: ( json_decode( $step['data'] ?? '[]', true ) ?: [] );
		$current_order = (int) $step['step_order'];

		$skip_next = false;

		switch ( $step_type ) {

			case 'email':
				$subject = $step_data['subject'] ?? ( $step['subject'] ?? '' );
				$body    = $step_data['content'] ?? ( $step['content'] ?? '' );

				$unsubscribe_url = ContactService::generate_unsubscribe_url( $contact_id, $contact['email'] );

				$subject = CampaignService::replace_placeholders( $subject, $contact, $unsubscribe_url );
				$body    = CampaignService::replace_placeholders( $body,    $contact, $unsubscribe_url );

				$from_name  = $step_data['from_name']  ?? get_bloginfo( 'name' );
				$from_email = $step_data['from_email'] ?? get_option( 'admin_email' );

				MailManager::send( [
					'to'         => $contact['email'],
					'subject'    => $subject,
					'body'       => $body,
					'from_name'  => $from_name,
					'from_email' => $from_email,
					'headers'    => [
						'List-Unsubscribe' => '<' . $unsubscribe_url . '>',
					],
				] );
				break;

			case 'wait':
				// The delay was applied at enqueue time; nothing to execute here.
				break;

			case 'tag':
				$tag_id = isset( $step_data['tag_id'] ) ? (int) $step_data['tag_id'] : 0;
				if ( $tag_id > 0 ) {
					Contact::add_tag( $contact_id, $tag_id );
				}
				break;

			case 'untag':
				$tag_id = isset( $step_data['tag_id'] ) ? (int) $step_data['tag_id'] : 0;
				if ( $tag_id > 0 ) {
					Contact::remove_tag( $contact_id, $tag_id );
				}
				break;

			case 'condition':
				// Simple equality condition. If it fails, skip the next step.
				$field    = $step_data['field']    ?? '';
				$operator = $step_data['operator'] ?? '=';
				$value    = $step_data['value']    ?? '';

				$contact_value = $contact[ $field ] ?? null;

				$passes = self::evaluate_condition( $contact_value, $operator, $value );

				if ( ! $passes ) {
					// Branch: skip the immediate next step and go to the one after.
					$skip_next = true;
				}
				break;

			default:
				// Unknown step type — mark done and move on.
				break;
		}

		// Mark queue item as done.
		Automation::update_queue_item( $queue_item_id, [
			'status'        => 'done',
			'processed_at'  => current_time( 'mysql' ),
			'updated_at'    => current_time( 'mysql' ),
		] );

		// Enqueue the next step (or the one after if we're skipping on a failed condition).
		if ( $skip_next ) {
			// Skip the next step: advance order by 2.
			self::enqueue_next_step( $automation_id, $current_order + 1, $contact_id );
		} else {
			self::enqueue_next_step( $automation_id, $current_order, $contact_id );
		}
	}

	/**
	 * Evaluate a simple condition comparison.
	 *
	 * @param mixed  $contact_value  The contact field value.
	 * @param string $operator       One of: =, !=, >, <, contains, not_contains.
	 * @param mixed  $expected       The expected value from the step configuration.
	 * @return bool
	 */
	private static function evaluate_condition( $contact_value, string $operator, $expected ): bool {
		switch ( $operator ) {
			case '=':
			case 'equals':
				return (string) $contact_value === (string) $expected;

			case '!=':
			case 'not_equals':
				return (string) $contact_value !== (string) $expected;

			case '>':
				return is_numeric( $contact_value ) && is_numeric( $expected )
					&& (float) $contact_value > (float) $expected;

			case '<':
				return is_numeric( $contact_value ) && is_numeric( $expected )
					&& (float) $contact_value < (float) $expected;

			case 'contains':
				return str_contains( (string) $contact_value, (string) $expected );

			case 'not_contains':
				return ! str_contains( (string) $contact_value, (string) $expected );

			default:
				return false;
		}
	}

	/**
	 * Find the next step after $current_step_order and enqueue it for the contact.
	 *
	 * If no further step exists, nothing is enqueued.
	 *
	 * @param int $automation_id
	 * @param int $current_step_order  The order index of the step that just ran.
	 * @param int $contact_id
	 * @return void
	 */
	public static function enqueue_next_step( int $automation_id, int $current_step_order, int $contact_id ): void {
		$next_step = Automation::get_next_step( $automation_id, $current_step_order );

		if ( ! $next_step ) {
			// No more steps — automation is complete for this contact.
			return;
		}

		$run_at = self::calculate_run_at( (int) $next_step['delay_value'], $next_step['delay_unit'] );

		Automation::enqueue( [
			'automation_id' => $automation_id,
			'step_id'       => (int) $next_step['id'],
			'contact_id'    => $contact_id,
			'status'        => 'pending',
			'run_at'        => $run_at,
			'created_at'    => current_time( 'mysql' ),
		] );
	}

	/**
	 * Calculate the MySQL datetime at which a step should run.
	 *
	 * A delay_value of 0 (regardless of unit) results in NOW() so the step
	 * is picked up on the very next cron pass.
	 *
	 * @param int    $delay_value  Numeric quantity.
	 * @param string $delay_unit   One of: minutes, hours, days, weeks.
	 * @return string  MySQL-formatted datetime.
	 */
	private static function calculate_run_at( int $delay_value, string $delay_unit ): string {
		if ( $delay_value <= 0 ) {
			return current_time( 'mysql' );
		}

		$seconds_map = [
			'minutes' => MINUTE_IN_SECONDS,
			'hours'   => HOUR_IN_SECONDS,
			'days'    => DAY_IN_SECONDS,
			'weeks'   => WEEK_IN_SECONDS,
		];

		$multiplier = $seconds_map[ $delay_unit ] ?? MINUTE_IN_SECONDS;
		$offset     = $delay_value * $multiplier;

		return gmdate( 'Y-m-d H:i:s', time() + $offset );
	}
}
