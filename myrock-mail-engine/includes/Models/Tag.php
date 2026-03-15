<?php
namespace MyRock\MailEngine\Models;
defined( 'ABSPATH' ) || exit;


class Tag {

	/**
	 * Returns the tags table name.
	 */
	private static function table(): string {
		global $wpdb;
		return $wpdb->prefix . 'mrme_tags';
	}

	/**
	 * Retrieve all tags ordered by name.
	 *
	 * @return array
	 */
	public static function all(): array {
		global $wpdb;

		$results = $wpdb->get_results(
			'SELECT * FROM ' . self::table() . ' ORDER BY name ASC',
			ARRAY_A
		);

		return $results ?: [];
	}

	/**
	 * Find a single tag by ID.
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
	 * Find a single tag by its slug.
	 *
	 * @param string $slug
	 * @return array|null
	 */
	public static function find_by_slug( string $slug ): ?array {
		global $wpdb;

		$row = $wpdb->get_row(
			$wpdb->prepare(
				'SELECT * FROM ' . self::table() . ' WHERE slug = %s LIMIT 1',
				$slug
			),
			ARRAY_A
		);

		return $row ?: null;
	}

	/**
	 * Create a new tag.
	 *
	 * @param array $data  Expected keys: name, [slug], [color]
	 * @return int|false  Inserted ID or false on failure.
	 */
	public static function create( array $data ): int|false {
		global $wpdb;

		$defaults = [
			'color' => '#BFA26A',
		];

		$data = array_merge( $defaults, $data );

		// Auto-generate slug if not provided.
		if ( empty( $data['slug'] ) ) {
			if ( empty( $data['name'] ) ) {
				return false;
			}
			$data['slug'] = self::generate_slug( $data['name'] );
		} else {
			// Ensure supplied slug is unique.
			$data['slug'] = self::generate_slug( $data['slug'] );
		}

		$result = $wpdb->insert( self::table(), $data );

		if ( false === $result ) {
			return false;
		}

		return (int) $wpdb->insert_id;
	}

	/**
	 * Update an existing tag.
	 *
	 * @param int   $id
	 * @param array $data
	 * @return bool
	 */
	public static function update( int $id, array $data ): bool {
		global $wpdb;

		// If renaming and no explicit slug supplied, regenerate slug.
		if ( isset( $data['name'] ) && ! isset( $data['slug'] ) ) {
			$data['slug'] = self::generate_slug( $data['name'], $id );
		} elseif ( isset( $data['slug'] ) ) {
			$data['slug'] = self::generate_slug( $data['slug'], $id );
		}

		$result = $wpdb->update(
			self::table(),
			$data,
			[ 'id' => $id ]
		);

		return false !== $result;
	}

	/**
	 * Delete a tag by ID.
	 *
	 * Also removes all contact-tag pivot rows for this tag.
	 *
	 * @param int $id
	 * @return bool
	 */
	public static function delete( int $id ): bool {
		global $wpdb;

		// Remove pivot associations.
		$wpdb->delete(
			$wpdb->prefix . 'mrme_contact_tags',
			[ 'tag_id' => $id ]
		);

		$result = $wpdb->delete( self::table(), [ 'id' => $id ] );

		return false !== $result;
	}

	/**
	 * Generate a unique slug for a tag name.
	 *
	 * Appends an incrementing numeric suffix (-2, -3, …) until a unique slug
	 * is found, optionally excluding a given tag ID (useful on update).
	 *
	 * @param string   $name        The tag name (or base slug).
	 * @param int|null $exclude_id  Tag ID to exclude from the uniqueness check.
	 * @return string
	 */
	public static function generate_slug( string $name, int $exclude_id = null ): string {
		global $wpdb;

		$base_slug = sanitize_title( $name );
		$slug      = $base_slug;
		$counter   = 2;
		$table     = self::table();

		while ( true ) {
			if ( null !== $exclude_id ) {
				$existing = $wpdb->get_var(
					$wpdb->prepare(
						"SELECT id FROM {$table} WHERE slug = %s AND id != %d LIMIT 1",
						$slug,
						$exclude_id
					)
				);
			} else {
				$existing = $wpdb->get_var(
					$wpdb->prepare(
						"SELECT id FROM {$table} WHERE slug = %s LIMIT 1",
						$slug
					)
				);
			}

			if ( null === $existing ) {
				break;
			}

			$slug = $base_slug . '-' . $counter;
			$counter++;
		}

		return $slug;
	}
}
