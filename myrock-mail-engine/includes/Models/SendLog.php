<?php
namespace MyRock\MailEngine\Models;
defined( 'ABSPATH' ) || exit;


class SendLog {

	/**
	 * Returns the send_logs table name.
	 */
	private static function table(): string {
		global $wpdb;
		return $wpdb->prefix . 'mrme_send_logs';
	}

	/**
	 * Create a new send log row.
	 *
	 * @param array $data  Expected keys: campaign_id, contact_id, email, [status], [sent_at], [error]
	 * @return int|false  Inserted ID or false on failure.
	 */
	public static function create( array $data ): int|false {
		global $wpdb;

		$defaults = [
			'status'  => 'pending',
			'sent_at' => null,
			'error'   => '',
		];

		$data = array_merge( $defaults, $data );

		$result = $wpdb->insert( self::table(), $data );

		if ( false === $result ) {
			return false;
		}

		return (int) $wpdb->insert_id;
	}

	/**
	 * Update the status (and optionally error) of a send log row.
	 *
	 * When setting status to 'sent', sent_at is automatically stamped.
	 *
	 * @param int    $id
	 * @param string $status  One of: pending, sent, failed, bounced
	 * @param string $error   Optional error message (stored when status is failed/bounced).
	 * @return void
	 */
	public static function update_status( int $id, string $status, string $error = '' ): void {
		global $wpdb;

		$data = [ 'status' => $status ];

		if ( 'sent' === $status ) {
			$data['sent_at'] = current_time( 'mysql' );
		}

		if ( '' !== $error ) {
			$data['error'] = $error;
		}

		$wpdb->update(
			self::table(),
			$data,
			[ 'id' => $id ]
		);
	}

	/**
	 * Find send log rows for a campaign with optional filtering.
	 *
	 * Supported args:
	 *   status  string  (pending|sent|failed|bounced)
	 *   limit   int     (default: 50)
	 *   offset  int     (default: 0)
	 *
	 * @param int   $campaign_id
	 * @param array $args
	 * @return array
	 */
	public static function find_by_campaign( int $campaign_id, array $args = [] ): array {
		global $wpdb;

		$defaults = [
			'status' => '',
			'limit'  => 50,
			'offset' => 0,
		];

		$args = wp_parse_args( $args, $defaults );

		$where  = [ 'campaign_id = %d' ];
		$params = [ $campaign_id ];

		if ( ! empty( $args['status'] ) ) {
			$where[]  = 'status = %s';
			$params[] = $args['status'];
		}

		$where_sql = 'WHERE ' . implode( ' AND ', $where );

		$limit  = max( 1, (int) $args['limit'] );
		$offset = max( 0, (int) $args['offset'] );

		$params[] = $limit;
		$params[] = $offset;

		$sql = 'SELECT * FROM ' . self::table() . " {$where_sql} ORDER BY id ASC LIMIT %d OFFSET %d";

		$results = $wpdb->get_results(
			$wpdb->prepare( $sql, $params ),
			ARRAY_A
		);

		return $results ?: [];
	}

	/**
	 * Count send log rows for a campaign, optionally filtered by status.
	 *
	 * @param int    $campaign_id
	 * @param string $status  Empty string means count all statuses.
	 * @return int
	 */
	public static function count_by_campaign( int $campaign_id, string $status = '' ): int {
		global $wpdb;

		if ( '' !== $status ) {
			$count = (int) $wpdb->get_var(
				$wpdb->prepare(
					'SELECT COUNT(*) FROM ' . self::table() . ' WHERE campaign_id = %d AND status = %s',
					$campaign_id,
					$status
				)
			);
		} else {
			$count = (int) $wpdb->get_var(
				$wpdb->prepare(
					'SELECT COUNT(*) FROM ' . self::table() . ' WHERE campaign_id = %d',
					$campaign_id
				)
			);
		}

		return $count;
	}

	/**
	 * Check whether a given contact has already been sent a specific campaign.
	 *
	 * A row with any status other than 'pending' is considered as already sent.
	 * 'pending' rows are treated as in-flight and will also return true so the
	 * same contact is never queued twice for the same campaign.
	 *
	 * @param int $campaign_id
	 * @param int $contact_id
	 * @return bool
	 */
	public static function already_sent( int $campaign_id, int $contact_id ): bool {
		global $wpdb;

		$count = (int) $wpdb->get_var(
			$wpdb->prepare(
				'SELECT COUNT(*) FROM ' . self::table() . ' WHERE campaign_id = %d AND contact_id = %d LIMIT 1',
				$campaign_id,
				$contact_id
			)
		);

		return $count > 0;
	}

	/**
	 * Bulk-create pending send log rows for a list of contacts.
	 *
	 * Contacts that have already been queued/sent for this campaign are skipped.
	 * Inserts are batched for performance.
	 *
	 * @param int   $campaign_id
	 * @param array $contacts    Array of contact rows (each must have 'id' and 'email' keys).
	 * @return void
	 */
	public static function bulk_create( int $campaign_id, array $contacts ): void {
		global $wpdb;

		if ( empty( $contacts ) ) {
			return;
		}

		$table   = self::table();
		$now     = current_time( 'mysql' );

		// Fetch contact IDs already logged for this campaign to avoid duplicates.
		$existing_ids = $wpdb->get_col(
			$wpdb->prepare(
				"SELECT contact_id FROM {$table} WHERE campaign_id = %d",
				$campaign_id
			)
		);

		$existing_ids = array_map( 'intval', $existing_ids );

		// Build batched INSERT for new contacts only.
		$values      = [];
		$value_rows  = [];

		foreach ( $contacts as $contact ) {
			$contact_id = (int) ( $contact['id'] ?? 0 );

			if ( ! $contact_id || in_array( $contact_id, $existing_ids, true ) ) {
				continue;
			}

			$email = $contact['email'] ?? '';

			$value_rows[] = $wpdb->prepare(
				'(%d, %d, %s, %s, %s)',
				$campaign_id,
				$contact_id,
				$email,
				'pending',
				$now
			);
		}

		if ( empty( $value_rows ) ) {
			return;
		}

		// Insert in chunks of 500 rows to avoid hitting packet size limits.
		$chunks = array_chunk( $value_rows, 500 );

		foreach ( $chunks as $chunk ) {
			$sql = "INSERT INTO {$table} (campaign_id, contact_id, email, status, sent_at) VALUES "
				. implode( ', ', $chunk );

			$wpdb->query( $sql ); // phpcs:ignore WordPress.DB.PreparedSQL.NotPrepared -- Values are individually prepared above.
		}
	}
}
