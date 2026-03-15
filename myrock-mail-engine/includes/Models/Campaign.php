<?php
namespace MyRock\MailEngine\Models;
defined( 'ABSPATH' ) || exit;


class Campaign {

	/**
	 * Returns the campaigns table name.
	 */
	private static function table(): string {
		global $wpdb;
		return $wpdb->prefix . 'mrme_campaigns';
	}

	/**
	 * Find a single campaign by ID.
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
	 * Retrieve campaigns with optional filtering.
	 *
	 * Supported args:
	 *   status   string  (draft|scheduled|sending|sent|cancelled)
	 *   search   string  (matches title)
	 *   orderby  string  (default: id)
	 *   order    string  (ASC|DESC, default: DESC)
	 *   limit    int     (default: 20)
	 *   offset   int     (default: 0)
	 *
	 * @param array $args
	 * @return array
	 */
	public static function all( array $args = [] ): array {
		global $wpdb;

		$defaults = [
			'status'  => '',
			'search'  => '',
			'orderby' => 'id',
			'order'   => 'DESC',
			'limit'   => 20,
			'offset'  => 0,
		];

		$args = wp_parse_args( $args, $defaults );

		$allowed_orderby = [ 'id', 'title', 'status', 'scheduled_at', 'sent_at', 'created_at', 'updated_at' ];
		$orderby = in_array( $args['orderby'], $allowed_orderby, true ) ? $args['orderby'] : 'id';
		$order   = strtoupper( $args['order'] ) === 'ASC' ? 'ASC' : 'DESC';

		$where  = [];
		$params = [];

		if ( ! empty( $args['status'] ) ) {
			$where[]  = 'status = %s';
			$params[] = $args['status'];
		}

		if ( ! empty( $args['search'] ) ) {
			$like     = '%' . $wpdb->esc_like( $args['search'] ) . '%';
			$where[]  = 'title LIKE %s';
			$params[] = $like;
		}

		$where_sql = $where ? ' WHERE ' . implode( ' AND ', $where ) : '';

		$limit  = max( 1, (int) $args['limit'] );
		$offset = max( 0, (int) $args['offset'] );

		$sql = 'SELECT * FROM ' . self::table() . "{$where_sql} ORDER BY {$orderby} {$order} LIMIT %d OFFSET %d";

		$params[] = $limit;
		$params[] = $offset;

		$results = $wpdb->get_results(
			$wpdb->prepare( $sql, $params ),
			ARRAY_A
		);

		return $results ?: [];
	}

	/**
	 * Count campaigns with optional filtering (same filters as all()).
	 *
	 * @param array $args
	 * @return int
	 */
	public static function count( array $args = [] ): int {
		global $wpdb;

		$defaults = [
			'status' => '',
			'search' => '',
		];

		$args = wp_parse_args( $args, $defaults );

		$where  = [];
		$params = [];

		if ( ! empty( $args['status'] ) ) {
			$where[]  = 'status = %s';
			$params[] = $args['status'];
		}

		if ( ! empty( $args['search'] ) ) {
			$like     = '%' . $wpdb->esc_like( $args['search'] ) . '%';
			$where[]  = 'title LIKE %s';
			$params[] = $like;
		}

		$where_sql = $where ? ' WHERE ' . implode( ' AND ', $where ) : '';

		$sql = 'SELECT COUNT(*) FROM ' . self::table() . $where_sql;

		if ( $params ) {
			$count = (int) $wpdb->get_var( $wpdb->prepare( $sql, $params ) );
		} else {
			$count = (int) $wpdb->get_var( $sql );
		}

		return $count;
	}

	/**
	 * Create a new campaign.
	 *
	 * @param array $data
	 * @return int|false  Inserted ID or false on failure.
	 */
	public static function create( array $data ): int|false {
		global $wpdb;

		$now = current_time( 'mysql' );

		$defaults = [
			'status'        => 'draft',
			'total_sent'    => 0,
			'total_opens'   => 0,
			'total_clicks'  => 0,
			'total_bounces' => 0,
			'total_unsubs'  => 0,
			'created_at'    => $now,
			'updated_at'    => $now,
		];

		$data = array_merge( $defaults, $data );

		$result = $wpdb->insert( self::table(), $data );

		if ( false === $result ) {
			return false;
		}

		return (int) $wpdb->insert_id;
	}

	/**
	 * Update an existing campaign.
	 *
	 * @param int   $id
	 * @param array $data
	 * @return bool
	 */
	public static function update( int $id, array $data ): bool {
		global $wpdb;

		$data['updated_at'] = current_time( 'mysql' );

		$result = $wpdb->update(
			self::table(),
			$data,
			[ 'id' => $id ]
		);

		return false !== $result;
	}

	/**
	 * Delete a campaign by ID.
	 *
	 * Also removes associated send log rows.
	 *
	 * @param int $id
	 * @return bool
	 */
	public static function delete( int $id ): bool {
		global $wpdb;

		// Remove send log rows.
		$wpdb->delete(
			$wpdb->prefix . 'mrme_send_logs',
			[ 'campaign_id' => $id ]
		);

		$result = $wpdb->delete( self::table(), [ 'id' => $id ] );

		return false !== $result;
	}

	/**
	 * Set the status of a campaign.
	 *
	 * @param int    $id
	 * @param string $status  One of: draft, scheduled, sending, sent, cancelled
	 * @return bool
	 */
	public static function set_status( int $id, string $status ): bool {
		$allowed = [ 'draft', 'scheduled', 'sending', 'sent', 'cancelled' ];

		if ( ! in_array( $status, $allowed, true ) ) {
			return false;
		}

		$data = [ 'status' => $status ];

		if ( 'sent' === $status ) {
			$data['sent_at'] = current_time( 'mysql' );
		}

		return self::update( $id, $data );
	}

	/**
	 * Atomically increment a numeric counter column on a campaign row.
	 *
	 * Supported fields: total_sent, total_opens, total_clicks, total_bounces, total_unsubs
	 *
	 * @param int    $id
	 * @param string $field
	 * @param int    $by
	 * @return void
	 */
	public static function increment( int $id, string $field, int $by = 1 ): void {
		global $wpdb;

		$allowed_fields = [ 'total_sent', 'total_opens', 'total_clicks', 'total_bounces', 'total_unsubs' ];

		if ( ! in_array( $field, $allowed_fields, true ) ) {
			return;
		}

		$table = self::table();

		$wpdb->query(
			$wpdb->prepare(
				"UPDATE {$table} SET {$field} = {$field} + %d, updated_at = %s WHERE id = %d",
				$by,
				current_time( 'mysql' ),
				$id
			)
		);
	}

	/**
	 * Get the list of recipient contacts for a campaign.
	 *
	 * Contacts are collected from list_ids and/or tag_ids stored on the campaign row.
	 * Duplicates (a contact in multiple lists/tags) are deduplicated.
	 * Only subscribed contacts are returned.
	 *
	 * @param int $campaign_id
	 * @return array  Array of contact rows.
	 */
	public static function get_recipients( int $campaign_id ): array {
		global $wpdb;

		$campaign = self::find( $campaign_id );

		if ( ! $campaign ) {
			return [];
		}

		$contacts_table = $wpdb->prefix . 'mrme_contacts';
		$contact_ids    = [];

		// Collect contacts from lists.
		if ( ! empty( $campaign['list_ids'] ) ) {
			$list_ids = array_filter( array_map( 'intval', explode( ',', $campaign['list_ids'] ) ) );

			if ( $list_ids ) {
				$placeholders = implode( ', ', array_fill( 0, count( $list_ids ), '%d' ) );
				$pivot_table  = $wpdb->prefix . 'mrme_contact_lists';

				$ids = $wpdb->get_col(
					$wpdb->prepare(
						"SELECT DISTINCT contact_id FROM {$pivot_table} WHERE list_id IN ({$placeholders})",
						$list_ids
					)
				);

				$contact_ids = array_merge( $contact_ids, array_map( 'intval', $ids ) );
			}
		}

		// Collect contacts from tags.
		if ( ! empty( $campaign['tag_ids'] ) ) {
			$tag_ids = array_filter( array_map( 'intval', explode( ',', $campaign['tag_ids'] ) ) );

			if ( $tag_ids ) {
				$placeholders = implode( ', ', array_fill( 0, count( $tag_ids ), '%d' ) );
				$pivot_table  = $wpdb->prefix . 'mrme_contact_tags';

				$ids = $wpdb->get_col(
					$wpdb->prepare(
						"SELECT DISTINCT contact_id FROM {$pivot_table} WHERE tag_id IN ({$placeholders})",
						$tag_ids
					)
				);

				$contact_ids = array_merge( $contact_ids, array_map( 'intval', $ids ) );
			}
		}

		if ( empty( $contact_ids ) ) {
			return [];
		}

		// Deduplicate and fetch subscribed contacts.
		$contact_ids  = array_values( array_unique( $contact_ids ) );
		$placeholders = implode( ', ', array_fill( 0, count( $contact_ids ), '%d' ) );

		$results = $wpdb->get_results(
			$wpdb->prepare(
				"SELECT * FROM {$contacts_table} WHERE id IN ({$placeholders}) AND status = 'subscribed' ORDER BY id ASC",
				$contact_ids
			),
			ARRAY_A
		);

		return $results ?: [];
	}

	/**
	 * Duplicate a campaign as a new draft.
	 *
	 * @param int $id  Source campaign ID.
	 * @return int|false  New campaign ID or false on failure.
	 */
	public static function duplicate( int $id ): int|false {
		$campaign = self::find( $id );

		if ( ! $campaign ) {
			return false;
		}

		// Remove identity and stat columns.
		unset( $campaign['id'], $campaign['sent_at'], $campaign['scheduled_at'] );

		$campaign['title']        = $campaign['title'] . ' (Copy)';
		$campaign['status']       = 'draft';
		$campaign['total_sent']   = 0;
		$campaign['total_opens']  = 0;
		$campaign['total_clicks'] = 0;
		$campaign['total_bounces'] = 0;
		$campaign['total_unsubs'] = 0;

		$now = current_time( 'mysql' );
		$campaign['created_at'] = $now;
		$campaign['updated_at'] = $now;

		return self::create( $campaign );
	}
}
