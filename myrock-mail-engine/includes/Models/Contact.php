<?php
namespace MyRock\MailEngine\Models;
defined( 'ABSPATH' ) || exit;


class Contact {

	/**
	 * Returns the contacts table name.
	 */
	private static function table(): string {
		global $wpdb;
		return $wpdb->prefix . 'mrme_contacts';
	}

	/**
	 * Returns the contact_list pivot table name.
	 */
	private static function list_table(): string {
		global $wpdb;
		return $wpdb->prefix . 'mrme_contact_lists';
	}

	/**
	 * Returns the contact_tag pivot table name.
	 */
	private static function tag_table(): string {
		global $wpdb;
		return $wpdb->prefix . 'mrme_contact_tags';
	}

	/**
	 * Find a single contact by ID.
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
	 * Find a single contact by email address.
	 *
	 * @param string $email
	 * @return array|null
	 */
	public static function find_by_email( string $email ): ?array {
		global $wpdb;
		$row = $wpdb->get_row(
			$wpdb->prepare(
				'SELECT * FROM ' . self::table() . ' WHERE email = %s LIMIT 1',
				$email
			),
			ARRAY_A
		);
		return $row ?: null;
	}

	/**
	 * Create a new contact.
	 *
	 * @param array $data
	 * @return int|false  Inserted ID or false on failure.
	 */
	public static function create( array $data ): int|false {
		global $wpdb;

		$now = current_time( 'mysql' );

		$defaults = [
			'status'     => 'pending',
			'created_at' => $now,
			'updated_at' => $now,
		];

		$data = array_merge( $defaults, $data );

		if ( isset( $data['meta'] ) && is_array( $data['meta'] ) ) {
			$data['meta'] = wp_json_encode( $data['meta'] );
		}

		$result = $wpdb->insert( self::table(), $data );

		if ( false === $result ) {
			return false;
		}

		return (int) $wpdb->insert_id;
	}

	/**
	 * Update an existing contact.
	 *
	 * @param int   $id
	 * @param array $data
	 * @return bool
	 */
	public static function update( int $id, array $data ): bool {
		global $wpdb;

		$data['updated_at'] = current_time( 'mysql' );

		if ( isset( $data['meta'] ) && is_array( $data['meta'] ) ) {
			$data['meta'] = wp_json_encode( $data['meta'] );
		}

		$result = $wpdb->update(
			self::table(),
			$data,
			[ 'id' => $id ]
		);

		return false !== $result;
	}

	/**
	 * Delete a contact by ID.
	 *
	 * @param int $id
	 * @return bool
	 */
	public static function delete( int $id ): bool {
		global $wpdb;

		// Remove pivot records first.
		$wpdb->delete( self::list_table(), [ 'contact_id' => $id ] );
		$wpdb->delete( self::tag_table(), [ 'contact_id' => $id ] );

		$result = $wpdb->delete( self::table(), [ 'id' => $id ] );

		return false !== $result;
	}

	/**
	 * Retrieve contacts with optional filtering.
	 *
	 * Supported args:
	 *   status   string
	 *   search   string  (matches email, first_name, last_name)
	 *   list_id  int
	 *   tag_id   int
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
			'list_id' => 0,
			'tag_id'  => 0,
			'orderby' => 'id',
			'order'   => 'DESC',
			'limit'   => 20,
			'offset'  => 0,
		];

		$args = wp_parse_args( $args, $defaults );

		$allowed_orderby = [ 'id', 'email', 'first_name', 'last_name', 'status', 'created_at', 'updated_at' ];
		$orderby = in_array( $args['orderby'], $allowed_orderby, true ) ? $args['orderby'] : 'id';
		$order   = strtoupper( $args['order'] ) === 'ASC' ? 'ASC' : 'DESC';

		$where  = [];
		$params = [];

		$contacts_table = self::table();

		// Build JOIN clauses conditionally.
		$join = '';

		if ( ! empty( $args['list_id'] ) ) {
			$list_table = self::list_table();
			$join      .= " INNER JOIN {$list_table} ON {$list_table}.contact_id = {$contacts_table}.id";
			$where[]    = "{$list_table}.list_id = %d";
			$params[]   = (int) $args['list_id'];
		}

		if ( ! empty( $args['tag_id'] ) ) {
			$tag_table = self::tag_table();
			$join     .= " INNER JOIN {$tag_table} ON {$tag_table}.contact_id = {$contacts_table}.id";
			$where[]   = "{$tag_table}.tag_id = %d";
			$params[]  = (int) $args['tag_id'];
		}

		if ( ! empty( $args['status'] ) ) {
			$where[]  = "{$contacts_table}.status = %s";
			$params[] = $args['status'];
		}

		if ( ! empty( $args['search'] ) ) {
			$like     = '%' . $wpdb->esc_like( $args['search'] ) . '%';
			$where[]  = "( {$contacts_table}.email LIKE %s OR {$contacts_table}.first_name LIKE %s OR {$contacts_table}.last_name LIKE %s )";
			$params[] = $like;
			$params[] = $like;
			$params[] = $like;
		}

		$where_sql = $where ? ' WHERE ' . implode( ' AND ', $where ) : '';

		$limit  = max( 1, (int) $args['limit'] );
		$offset = max( 0, (int) $args['offset'] );

		$sql = "SELECT DISTINCT {$contacts_table}.* FROM {$contacts_table}{$join}{$where_sql} ORDER BY {$contacts_table}.{$orderby} {$order} LIMIT %d OFFSET %d";

		$params[] = $limit;
		$params[] = $offset;

		$results = $wpdb->get_results(
			$wpdb->prepare( $sql, $params ),
			ARRAY_A
		);

		return $results ?: [];
	}

	/**
	 * Count contacts with optional filtering (same filters as all()).
	 *
	 * @param array $args
	 * @return int
	 */
	public static function count( array $args = [] ): int {
		global $wpdb;

		$defaults = [
			'status'  => '',
			'search'  => '',
			'list_id' => 0,
			'tag_id'  => 0,
		];

		$args = wp_parse_args( $args, $defaults );

		$where  = [];
		$params = [];

		$contacts_table = self::table();
		$join           = '';

		if ( ! empty( $args['list_id'] ) ) {
			$list_table = self::list_table();
			$join      .= " INNER JOIN {$list_table} ON {$list_table}.contact_id = {$contacts_table}.id";
			$where[]    = "{$list_table}.list_id = %d";
			$params[]   = (int) $args['list_id'];
		}

		if ( ! empty( $args['tag_id'] ) ) {
			$tag_table = self::tag_table();
			$join     .= " INNER JOIN {$tag_table} ON {$tag_table}.contact_id = {$contacts_table}.id";
			$where[]   = "{$tag_table}.tag_id = %d";
			$params[]  = (int) $args['tag_id'];
		}

		if ( ! empty( $args['status'] ) ) {
			$where[]  = "{$contacts_table}.status = %s";
			$params[] = $args['status'];
		}

		if ( ! empty( $args['search'] ) ) {
			$like     = '%' . $wpdb->esc_like( $args['search'] ) . '%';
			$where[]  = "( {$contacts_table}.email LIKE %s OR {$contacts_table}.first_name LIKE %s OR {$contacts_table}.last_name LIKE %s )";
			$params[] = $like;
			$params[] = $like;
			$params[] = $like;
		}

		$where_sql = $where ? ' WHERE ' . implode( ' AND ', $where ) : '';

		$sql = "SELECT COUNT(DISTINCT {$contacts_table}.id) FROM {$contacts_table}{$join}{$where_sql}";

		if ( $params ) {
			$count = (int) $wpdb->get_var( $wpdb->prepare( $sql, $params ) );
		} else {
			$count = (int) $wpdb->get_var( $sql );
		}

		return $count;
	}

	/**
	 * Subscribe a contact to a mailing list.
	 *
	 * Uses INSERT IGNORE so duplicate subscriptions are silently skipped.
	 *
	 * @param int $contact_id
	 * @param int $list_id
	 * @return void
	 */
	public static function subscribe_to_list( int $contact_id, int $list_id ): void {
		global $wpdb;

		$table = self::list_table();

		$wpdb->query(
			$wpdb->prepare(
				"INSERT IGNORE INTO {$table} (contact_id, list_id, created_at) VALUES (%d, %d, %s)",
				$contact_id,
				$list_id,
				current_time( 'mysql' )
			)
		);
	}

	/**
	 * Unsubscribe a contact from a mailing list.
	 *
	 * @param int $contact_id
	 * @param int $list_id
	 * @return void
	 */
	public static function unsubscribe_from_list( int $contact_id, int $list_id ): void {
		global $wpdb;

		$wpdb->delete(
			self::list_table(),
			[
				'contact_id' => $contact_id,
				'list_id'    => $list_id,
			]
		);
	}

	/**
	 * Get all mailing lists a contact belongs to.
	 *
	 * @param int $contact_id
	 * @return array  Array of list rows.
	 */
	public static function get_lists( int $contact_id ): array {
		global $wpdb;

		$lists_table   = $wpdb->prefix . 'mrme_lists';
		$pivot_table   = self::list_table();

		$results = $wpdb->get_results(
			$wpdb->prepare(
				"SELECT l.* FROM {$lists_table} l
				 INNER JOIN {$pivot_table} cl ON cl.list_id = l.id
				 WHERE cl.contact_id = %d
				 ORDER BY l.id ASC",
				$contact_id
			),
			ARRAY_A
		);

		return $results ?: [];
	}

	/**
	 * Get all tags assigned to a contact.
	 *
	 * @param int $contact_id
	 * @return array  Array of tag rows.
	 */
	public static function get_tags( int $contact_id ): array {
		global $wpdb;

		$tags_table  = $wpdb->prefix . 'mrme_tags';
		$pivot_table = self::tag_table();

		$results = $wpdb->get_results(
			$wpdb->prepare(
				"SELECT t.* FROM {$tags_table} t
				 INNER JOIN {$pivot_table} ct ON ct.tag_id = t.id
				 WHERE ct.contact_id = %d
				 ORDER BY t.id ASC",
				$contact_id
			),
			ARRAY_A
		);

		return $results ?: [];
	}

	/**
	 * Add a tag to a contact.
	 *
	 * Uses INSERT IGNORE so duplicate assignments are silently skipped.
	 *
	 * @param int $contact_id
	 * @param int $tag_id
	 * @return void
	 */
	public static function add_tag( int $contact_id, int $tag_id ): void {
		global $wpdb;

		$table = self::tag_table();

		$wpdb->query(
			$wpdb->prepare(
				"INSERT IGNORE INTO {$table} (contact_id, tag_id, created_at) VALUES (%d, %d, %s)",
				$contact_id,
				$tag_id,
				current_time( 'mysql' )
			)
		);
	}

	/**
	 * Remove a tag from a contact.
	 *
	 * @param int $contact_id
	 * @param int $tag_id
	 * @return void
	 */
	public static function remove_tag( int $contact_id, int $tag_id ): void {
		global $wpdb;

		$wpdb->delete(
			self::tag_table(),
			[
				'contact_id' => $contact_id,
				'tag_id'     => $tag_id,
			]
		);
	}

	/**
	 * Set the status of a contact.
	 *
	 * @param int    $contact_id
	 * @param string $status  One of: subscribed, unsubscribed, pending, bounced, complained
	 * @return bool
	 */
	public static function set_status( int $contact_id, string $status ): bool {
		$allowed = [ 'subscribed', 'unsubscribed', 'pending', 'bounced', 'complained' ];

		if ( ! in_array( $status, $allowed, true ) ) {
			return false;
		}

		return self::update( $contact_id, [ 'status' => $status ] );
	}
}
