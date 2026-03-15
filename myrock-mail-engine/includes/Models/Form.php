<?php
namespace MyRock\MailEngine\Models;
defined( 'ABSPATH' ) || exit;


class Form {

	/**
	 * Returns the forms table name.
	 */
	private static function table(): string {
		global $wpdb;
		return $wpdb->prefix . 'mrme_forms';
	}

	/**
	 * Find a single form by ID.
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
	 * Retrieve all forms ordered by ID descending.
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
	 * Create a new form.
	 *
	 * @param array $data
	 * @return int|false  Inserted ID or false on failure.
	 */
	public static function create( array $data ): int|false {
		global $wpdb;

		$defaults = [
			'status'          => 'active',
			'double_optin'    => 0,
			'submissions'     => 0,
			'success_message' => '',
			'redirect_url'    => '',
			'list_ids'        => '',
			'tag_ids'         => '',
			'fields'          => '[]',
			'created_at'      => current_time( 'mysql' ),
		];

		$data = array_merge( $defaults, $data );

		// Encode fields array to JSON if necessary.
		if ( isset( $data['fields'] ) && is_array( $data['fields'] ) ) {
			$data['fields'] = wp_json_encode( $data['fields'] );
		}

		$result = $wpdb->insert( self::table(), $data );

		if ( false === $result ) {
			return false;
		}

		return (int) $wpdb->insert_id;
	}

	/**
	 * Update an existing form.
	 *
	 * @param int   $id
	 * @param array $data
	 * @return bool
	 */
	public static function update( int $id, array $data ): bool {
		global $wpdb;

		if ( isset( $data['fields'] ) && is_array( $data['fields'] ) ) {
			$data['fields'] = wp_json_encode( $data['fields'] );
		}

		$result = $wpdb->update(
			self::table(),
			$data,
			[ 'id' => $id ]
		);

		return false !== $result;
	}

	/**
	 * Delete a form by ID.
	 *
	 * @param int $id
	 * @return bool
	 */
	public static function delete( int $id ): bool {
		global $wpdb;

		$result = $wpdb->delete( self::table(), [ 'id' => $id ] );

		return false !== $result;
	}

	/**
	 * Atomically increment the submission counter for a form.
	 *
	 * @param int $id
	 * @return void
	 */
	public static function increment_submissions( int $id ): void {
		global $wpdb;

		$table = self::table();

		$wpdb->query(
			$wpdb->prepare(
				"UPDATE {$table} SET submissions = submissions + 1 WHERE id = %d",
				$id
			)
		);
	}

	/**
	 * Get the decoded fields array for a form.
	 *
	 * @param int $id
	 * @return array  Empty array if the form does not exist or fields are invalid JSON.
	 */
	public static function get_fields( int $id ): array {
		$form = self::find( $id );

		if ( ! $form || empty( $form['fields'] ) ) {
			return [];
		}

		$decoded = json_decode( $form['fields'], true );

		return is_array( $decoded ) ? $decoded : [];
	}

	/**
	 * Get the list IDs associated with a form as an array of integers.
	 *
	 * @param int $id
	 * @return int[]
	 */
	public static function get_list_ids( int $id ): array {
		$form = self::find( $id );

		if ( ! $form || empty( $form['list_ids'] ) ) {
			return [];
		}

		return array_values(
			array_filter(
				array_map( 'intval', explode( ',', $form['list_ids'] ) )
			)
		);
	}

	/**
	 * Get the tag IDs associated with a form as an array of integers.
	 *
	 * @param int $id
	 * @return int[]
	 */
	public static function get_tag_ids( int $id ): array {
		$form = self::find( $id );

		if ( ! $form || empty( $form['tag_ids'] ) ) {
			return [];
		}

		return array_values(
			array_filter(
				array_map( 'intval', explode( ',', $form['tag_ids'] ) )
			)
		);
	}
}
