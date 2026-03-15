<?php
namespace MyRock\MailEngine\Models;
defined( 'ABSPATH' ) || exit;


class Automation {

	/**
	 * Returns the automations table name.
	 */
	private static function table(): string {
		global $wpdb;
		return $wpdb->prefix . 'mrme_automations';
	}

	/**
	 * Returns the automation_steps table name.
	 */
	private static function steps_table(): string {
		global $wpdb;
		return $wpdb->prefix . 'mrme_automation_steps';
	}

	/**
	 * Returns the automation_queue table name.
	 */
	private static function queue_table(): string {
		global $wpdb;
		return $wpdb->prefix . 'mrme_automation_queue';
	}

	// -------------------------------------------------------------------------
	// Automation CRUD
	// -------------------------------------------------------------------------

	/**
	 * Find a single automation by ID.
	 *
	 * @param int $id
	 * @return array|null
	 */
	public static function find( int $id ): ?array {
		global $wpdb;

		$row = $wpdb->get_row(
			$wpdb->prepare(
				'SELECT * FROM ' . self::table() . ' WHERE id = %d LIMIT 1',
				$id
			),
			ARRAY_A
		);

		return $row ?: null;
	}

	/**
	 * Retrieve all automations ordered by ID descending.
	 *
	 * @return array
	 */
	public static function all(): array {
		global $wpdb;

		$results = $wpdb->get_results(
			'SELECT * FROM ' . self::table() . ' ORDER BY id DESC',
			ARRAY_A
		);

		return $results ?: [];
	}

	/**
	 * Create a new automation.
	 *
	 * @param array $data  Expected keys: name, trigger, [trigger_data], [status]
	 * @return int|false  Inserted ID or false on failure.
	 */
	public static function create( array $data ): int|false {
		global $wpdb;

		$defaults = [
			'status'       => 'inactive',
			'trigger_data' => '{}',
			'created_at'   => current_time( 'mysql' ),
		];

		$data = array_merge( $defaults, $data );

		if ( isset( $data['trigger_data'] ) && is_array( $data['trigger_data'] ) ) {
			$data['trigger_data'] = wp_json_encode( $data['trigger_data'] );
		}

		$result = $wpdb->insert( self::table(), $data );

		if ( false === $result ) {
			return false;
		}

		return (int) $wpdb->insert_id;
	}

	/**
	 * Update an existing automation.
	 *
	 * @param int   $id
	 * @param array $data
	 * @return bool
	 */
	public static function update( int $id, array $data ): bool {
		global $wpdb;

		if ( isset( $data['trigger_data'] ) && is_array( $data['trigger_data'] ) ) {
			$data['trigger_data'] = wp_json_encode( $data['trigger_data'] );
		}

		$result = $wpdb->update(
			self::table(),
			$data,
			[ 'id' => $id ]
		);

		return false !== $result;
	}

	/**
	 * Delete an automation and all associated steps and queue entries.
	 *
	 * @param int $id
	 * @return bool
	 */
	public static function delete( int $id ): bool {
		global $wpdb;

		// Remove queue entries first.
		$wpdb->delete( self::queue_table(), [ 'automation_id' => $id ] );

		// Remove steps.
		$wpdb->delete( self::steps_table(), [ 'automation_id' => $id ] );

		// Remove the automation itself.
		$result = $wpdb->delete( self::table(), [ 'id' => $id ] );

		return false !== $result;
	}

	// -------------------------------------------------------------------------
	// Steps
	// -------------------------------------------------------------------------

	/**
	 * Get all steps for an automation, ordered by step_order ascending.
	 *
	 * @param int $automation_id
	 * @return array
	 */
	public static function get_steps( int $automation_id ): array {
		global $wpdb;

		$table   = self::steps_table();
		$results = $wpdb->get_results(
			$wpdb->prepare(
				"SELECT * FROM {$table} WHERE automation_id = %d ORDER BY step_order ASC",
				$automation_id
			),
			ARRAY_A
		);

		return $results ?: [];
	}

	/**
	 * Replace all steps for an automation.
	 *
	 * Deletes existing steps then inserts the supplied array. Each step in the
	 * array should have: type, step_order, [delay_value], [delay_unit], [data].
	 *
	 * @param int   $automation_id
	 * @param array $steps
	 * @return void
	 */
	public static function save_steps( int $automation_id, array $steps ): void {
		global $wpdb;

		$table = self::steps_table();

		// Delete existing steps (and their queue entries).
		$existing_step_ids = $wpdb->get_col(
			$wpdb->prepare(
				"SELECT id FROM {$table} WHERE automation_id = %d",
				$automation_id
			)
		);

		if ( $existing_step_ids ) {
			foreach ( $existing_step_ids as $step_id ) {
				$wpdb->delete( self::queue_table(), [ 'step_id' => (int) $step_id ] );
			}
		}

		$wpdb->delete( $table, [ 'automation_id' => $automation_id ] );

		// Insert the new steps.
		foreach ( $steps as $order => $step ) {
			$row = [
				'automation_id' => $automation_id,
				'step_order'    => isset( $step['step_order'] ) ? (int) $step['step_order'] : (int) $order,
				'type'          => $step['type'] ?? 'email',
				'delay_value'   => isset( $step['delay_value'] ) ? (int) $step['delay_value'] : 0,
				'delay_unit'    => $step['delay_unit'] ?? 'minutes',
				'data'          => isset( $step['data'] ) && is_array( $step['data'] )
									? wp_json_encode( $step['data'] )
									: ( $step['data'] ?? '{}' ),
			];

			$wpdb->insert( $table, $row );
		}
	}

	// -------------------------------------------------------------------------
	// Trigger helpers
	// -------------------------------------------------------------------------

	/**
	 * Get all active automations that match a given trigger.
	 *
	 * @param string $trigger
	 * @return array
	 */
	public static function get_active_by_trigger( string $trigger ): array {
		global $wpdb;

		$table   = self::table();
		$results = $wpdb->get_results(
			$wpdb->prepare(
				"SELECT * FROM {$table} WHERE trigger = %s AND status = 'active' ORDER BY id ASC",
				$trigger
			),
			ARRAY_A
		);

		return $results ?: [];
	}

	// -------------------------------------------------------------------------
	// Queue management
	// -------------------------------------------------------------------------

	/**
	 * Add an item to the automation queue.
	 *
	 * @param int    $automation_id
	 * @param int    $step_id
	 * @param int    $contact_id
	 * @param string $run_at  MySQL datetime string (e.g. '2026-03-15 14:00:00').
	 * @return void
	 */
	public static function enqueue( int $automation_id, int $step_id, int $contact_id, string $run_at ): void {
		global $wpdb;

		$wpdb->insert(
			self::queue_table(),
			[
				'automation_id' => $automation_id,
				'step_id'       => $step_id,
				'contact_id'    => $contact_id,
				'run_at'        => $run_at,
				'status'        => 'pending',
				'created_at'    => current_time( 'mysql' ),
			]
		);
	}

	/**
	 * Get pending queue items whose run_at time has passed.
	 *
	 * Joins on steps and automations so callers have all data in one query.
	 *
	 * @param int $limit  Maximum rows to return (default 50).
	 * @return array
	 */
	public static function get_pending_queue_items( int $limit = 50 ): array {
		global $wpdb;

		$queue_table      = self::queue_table();
		$steps_table      = self::steps_table();
		$automations_table = self::table();

		$results = $wpdb->get_results(
			$wpdb->prepare(
				"SELECT q.*, s.type AS step_type, s.data AS step_data, s.step_order,
				        a.name AS automation_name, a.trigger AS automation_trigger
				 FROM {$queue_table} q
				 INNER JOIN {$steps_table} s ON s.id = q.step_id
				 INNER JOIN {$automations_table} a ON a.id = q.automation_id
				 WHERE q.status = 'pending'
				   AND q.run_at <= %s
				 ORDER BY q.run_at ASC
				 LIMIT %d",
				current_time( 'mysql' ),
				$limit
			),
			ARRAY_A
		);

		return $results ?: [];
	}

	/**
	 * Mark a queue item as done.
	 *
	 * @param int $queue_id
	 * @return void
	 */
	public static function mark_queue_done( int $queue_id ): void {
		global $wpdb;

		$wpdb->update(
			self::queue_table(),
			[ 'status' => 'done' ],
			[ 'id' => $queue_id ]
		);
	}

	/**
	 * Mark a queue item as failed, optionally storing an error message.
	 *
	 * The queue table's data JSON column is reused to store the error string
	 * (stored as {"error":"…"}) so no schema change is required.
	 *
	 * @param int    $queue_id
	 * @param string $error
	 * @return void
	 */
	public static function mark_queue_failed( int $queue_id, string $error = '' ): void {
		global $wpdb;

		$update = [ 'status' => 'failed' ];

		$wpdb->update(
			self::queue_table(),
			$update,
			[ 'id' => $queue_id ]
		);

		// Store the error message in the steps data if provided.
		if ( '' !== $error ) {
			$queue_table = self::queue_table();
			$wpdb->query(
				$wpdb->prepare(
					"UPDATE {$queue_table} SET error = %s WHERE id = %d",
					$error,
					$queue_id
				)
			);
		}
	}
}
