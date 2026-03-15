<?php
namespace MyRock\MailEngine\Services;

defined( 'ABSPATH' ) || exit;

use MyRock\MailEngine\Models\Campaign;
use MyRock\MailEngine\Models\SendLog;
use MyRock\MailEngine\Mail\MailManager;

class CampaignService {

	/**
	 * Schedule a campaign to send at a specific datetime.
	 *
	 * Only campaigns that are in 'draft' or 'scheduled' status can be scheduled.
	 *
	 * @param int    $campaign_id
	 * @param string $datetime  MySQL-formatted datetime (e.g. '2026-03-15 14:00:00').
	 * @return bool
	 */
	public static function schedule( int $campaign_id, string $datetime ): bool {
		$campaign = Campaign::find( $campaign_id );

		if ( ! $campaign ) {
			return false;
		}

		if ( ! in_array( $campaign['status'], [ 'draft', 'scheduled' ], true ) ) {
			return false;
		}

		return Campaign::update( $campaign_id, [
			'status'       => 'scheduled',
			'scheduled_at' => $datetime,
		] );
	}

	/**
	 * Immediately send a campaign.
	 *
	 * @param int $campaign_id
	 * @return bool
	 */
	public static function send_now( int $campaign_id ): bool {
		return self::process_campaign( $campaign_id );
	}

	/**
	 * Process all campaigns whose scheduled_at time has passed.
	 *
	 * Intended to be called by a WP-Cron hook every 5 minutes.
	 *
	 * @return void
	 */
	public static function process_scheduled(): void {
		$campaigns = Campaign::all( [
			'status'              => 'scheduled',
			'scheduled_at_before' => current_time( 'mysql' ),
			'limit'               => 50,
			'offset'              => 0,
		] );

		foreach ( $campaigns as $campaign ) {
			self::process_campaign( (int) $campaign['id'] );
		}
	}

	/**
	 * Kick off sending for a campaign.
	 *
	 * Sets status to 'sending', builds the send-log queue, then fires the
	 * first batch immediately.
	 *
	 * @param int $campaign_id
	 * @return bool
	 */
	public static function process_campaign( int $campaign_id ): bool {
		$campaign = Campaign::find( $campaign_id );

		if ( ! $campaign ) {
			return false;
		}

		// Mark as sending.
		Campaign::update( $campaign_id, [ 'status' => 'sending' ] );

		// Gather recipients.
		$recipients = Campaign::get_recipients( $campaign_id );

		if ( empty( $recipients ) ) {
			// No recipients — mark sent immediately.
			Campaign::update( $campaign_id, [
				'status'  => 'sent',
				'sent_at' => current_time( 'mysql' ),
			] );
			return true;
		}

		// Build pending send-log rows.
		$log_rows = [];
		foreach ( $recipients as $contact ) {
			$log_rows[] = [
				'campaign_id' => $campaign_id,
				'contact_id'  => (int) $contact['id'],
				'email'       => $contact['email'],
				'status'      => 'pending',
			];
		}

		SendLog::bulk_create( $log_rows );

		// Send the first batch right away.
		self::send_batch( $campaign_id, 50 );

		return true;
	}

	/**
	 * Send one batch of pending emails for a campaign.
	 *
	 * When the queue is fully drained the campaign is marked 'sent'.
	 *
	 * @param int $campaign_id
	 * @param int $batch_size  Number of emails to send in this call.
	 * @return int             Number of emails attempted in this batch.
	 */
	public static function send_batch( int $campaign_id, int $batch_size = 50 ): int {
		$campaign = Campaign::find( $campaign_id );

		if ( ! $campaign ) {
			return 0;
		}

		$pending_logs = SendLog::get_pending_for_campaign( $campaign_id, $batch_size );

		if ( empty( $pending_logs ) ) {
			// All done — nothing left to send.
			Campaign::update( $campaign_id, [
				'status'  => 'sent',
				'sent_at' => current_time( 'mysql' ),
			] );
			return 0;
		}

		$attempted = 0;

		foreach ( $pending_logs as $log ) {
			$contact_id = (int) $log['contact_id'];

			// Build contact context for placeholder replacement.
			$contact = \MyRock\MailEngine\Models\Contact::find( $contact_id );
			if ( ! $contact ) {
				SendLog::update( (int) $log['id'], [ 'status' => 'failed', 'error' => 'Contact not found.' ] );
				$attempted++;
				continue;
			}

			$unsubscribe_url = ContactService::generate_unsubscribe_url( $contact_id, $contact['email'] );

			$subject = self::replace_placeholders( $campaign['subject'] ?? '', $contact, $unsubscribe_url );
			$body    = self::replace_placeholders( $campaign['content'] ?? '', $contact, $unsubscribe_url );

			$sent = MailManager::send( [
				'to'      => $contact['email'],
				'subject' => $subject,
				'body'    => $body,
				'headers' => [
					'List-Unsubscribe' => '<' . $unsubscribe_url . '>',
				],
			] );

			if ( $sent ) {
				SendLog::update( (int) $log['id'], [
					'status' => 'sent',
					'sent_at' => current_time( 'mysql' ),
				] );
			} else {
				SendLog::update( (int) $log['id'], [
					'status' => 'failed',
					'error'  => 'MailManager::send() returned false.',
				] );
			}

			$attempted++;
		}

		// Increment campaign total_sent counter.
		Campaign::increment_sent( $campaign_id, $attempted );

		// Check whether all logs are now exhausted.
		$remaining = SendLog::count_pending_for_campaign( $campaign_id );
		if ( $remaining === 0 ) {
			Campaign::update( $campaign_id, [
				'status'  => 'sent',
				'sent_at' => current_time( 'mysql' ),
			] );
		}

		return $attempted;
	}

	/**
	 * Send a test copy of a campaign to a single address.
	 *
	 * No send-log records are created. Placeholders are replaced with dummy data.
	 *
	 * @param int    $campaign_id
	 * @param string $to_email
	 * @return bool
	 */
	public static function send_test( int $campaign_id, string $to_email ): bool {
		$campaign = Campaign::find( $campaign_id );

		if ( ! $campaign || ! is_email( $to_email ) ) {
			return false;
		}

		$dummy_contact = [
			'first_name' => 'Test',
			'last_name'  => 'User',
			'email'      => sanitize_email( $to_email ),
			'company'    => 'Test Company',
		];

		$subject = self::replace_placeholders( $campaign['subject'] ?? '', $dummy_contact, home_url( '/' ) );
		$body    = self::replace_placeholders( $campaign['content'] ?? '', $dummy_contact, home_url( '/' ) );

		return MailManager::send( [
			'to'      => $to_email,
			'subject' => '[TEST] ' . $subject,
			'body'    => $body,
		] );
	}

	/**
	 * Replace merge-tag placeholders in a string with contact field values.
	 *
	 * Supported tags:
	 *   {{first_name}}, {{last_name}}, {{email}}, {{company}}, {{unsubscribe_url}}
	 *
	 * @param string $content
	 * @param array  $contact         Associative array with contact fields.
	 * @param string $unsubscribe_url
	 * @return string
	 */
	public static function replace_placeholders( string $content, array $contact, string $unsubscribe_url = '' ): string {
		$replacements = [
			'{{first_name}}'     => $contact['first_name'] ?? '',
			'{{last_name}}'      => $contact['last_name']  ?? '',
			'{{email}}'          => $contact['email']       ?? '',
			'{{company}}'        => $contact['company']     ?? '',
			'{{unsubscribe_url}}' => $unsubscribe_url,
		];

		return str_replace(
			array_keys( $replacements ),
			array_values( $replacements ),
			$content
		);
	}
}
