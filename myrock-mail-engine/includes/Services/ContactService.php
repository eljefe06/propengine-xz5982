<?php
namespace MyRock\MailEngine\Services;

defined( 'ABSPATH' ) || exit;

use MyRock\MailEngine\Models\Contact;

class ContactService {

	/**
	 * Retrieve or generate the plugin secret key stored in wp_options.
	 *
	 * @return string
	 */
	private static function get_secret_key(): string {
		$key = get_option( 'mrme_secret_key' );
		if ( ! $key ) {
			$key = wp_generate_password( 32, false );
			update_option( 'mrme_secret_key', $key, false );
		}
		return $key;
	}

	/**
	 * Sanitize an input array of contact fields.
	 *
	 * @param array $data Raw input data.
	 * @return array      Sanitized data (only known contact fields).
	 */
	private static function sanitize_fields( array $data ): array {
		$sanitized = [];

		if ( isset( $data['email'] ) ) {
			$sanitized['email'] = sanitize_email( $data['email'] );
		}

		if ( isset( $data['first_name'] ) ) {
			$sanitized['first_name'] = sanitize_text_field( $data['first_name'] );
		}

		if ( isset( $data['last_name'] ) ) {
			$sanitized['last_name'] = sanitize_text_field( $data['last_name'] );
		}

		if ( isset( $data['phone'] ) ) {
			$sanitized['phone'] = sanitize_text_field( $data['phone'] );
		}

		if ( isset( $data['company'] ) ) {
			$sanitized['company'] = sanitize_text_field( $data['company'] );
		}

		if ( isset( $data['status'] ) ) {
			$sanitized['status'] = sanitize_text_field( $data['status'] );
		}

		if ( isset( $data['meta'] ) ) {
			// Accept array or JSON string; store as-is (Contact model JSON-encodes arrays).
			$sanitized['meta'] = $data['meta'];
		}

		return $sanitized;
	}

	/**
	 * Create a contact or update it if the email already exists.
	 *
	 * Unlike create(), this method updates the stored fields when the email
	 * is already in the database instead of just syncing lists/tags.
	 *
	 * @param array $data Contact fields. Optional keys: list_ids (int[]), tag_ids (int[]).
	 * @return int|false  Contact ID on success, false on failure.
	 */
	public static function create_or_update( array $data ): int|false {
		if ( empty( $data['email'] ) || ! is_email( $data['email'] ) ) {
			return false;
		}

		$email    = sanitize_email( $data['email'] );
		$existing = Contact::find_by_email( $email );

		if ( $existing ) {
			$contact_id = (int) $existing['id'];
			self::update( $contact_id, $data );
			return $contact_id;
		}

		return self::create( $data );
	}

	/**
	 * Generate a signed token for a contact by type.
	 *
	 * Supported types:
	 *   'confirm_optin'  — double opt-in confirmation token
	 *   'unsubscribe'    — one-click unsubscribe token (default)
	 *
	 * @param int    $contact_id
	 * @param string $type  Token type identifier.
	 * @return string  Base64-encoded token, or empty string if contact not found.
	 */
	public static function generate_token( int $contact_id, string $type ): string {
		$contact = Contact::find( $contact_id );

		if ( ! $contact ) {
			return '';
		}

		$email = $contact['email'] ?? '';

		if ( 'confirm_optin' === $type ) {
			return self::build_confirm_token( $contact_id, $email );
		}

		return self::build_unsubscribe_token( $contact_id, $email );
	}

	/**
	 * Create a new contact.
	 *
	 * @param array $data  Contact fields. Optional keys: list_ids (int[]), tag_ids (int[]).
	 * @return int|false   New contact ID, existing contact ID if duplicate, or false on failure.
	 */
	public static function create( array $data ): int|false {
		// Validate email.
		if ( empty( $data['email'] ) || ! is_email( $data['email'] ) ) {
			return false;
		}

		$email = sanitize_email( $data['email'] );

		// Duplicate check — return existing ID if found.
		$existing = Contact::find_by_email( $email );
		if ( $existing ) {
			$contact_id = (int) $existing['id'];

			// Still subscribe / tag as requested even for existing contacts.
			if ( ! empty( $data['list_ids'] ) && is_array( $data['list_ids'] ) ) {
				foreach ( $data['list_ids'] as $list_id ) {
					self::subscribe( $contact_id, (int) $list_id );
				}
			}

			if ( ! empty( $data['tag_ids'] ) && is_array( $data['tag_ids'] ) ) {
				foreach ( $data['tag_ids'] as $tag_id ) {
					Contact::add_tag( $contact_id, (int) $tag_id );
				}
			}

			return $contact_id;
		}

		// Sanitize and strip non-column keys before insert.
		$sanitized = self::sanitize_fields( $data );
		$sanitized['email'] = $email;

		$contact_id = Contact::create( $sanitized );

		if ( false === $contact_id ) {
			return false;
		}

		// Subscribe to lists.
		if ( ! empty( $data['list_ids'] ) && is_array( $data['list_ids'] ) ) {
			foreach ( $data['list_ids'] as $list_id ) {
				self::subscribe( $contact_id, (int) $list_id );
			}
		}

		// Add tags.
		if ( ! empty( $data['tag_ids'] ) && is_array( $data['tag_ids'] ) ) {
			foreach ( $data['tag_ids'] as $tag_id ) {
				Contact::add_tag( $contact_id, (int) $tag_id );
			}
		}

		return $contact_id;
	}

	/**
	 * Update an existing contact.
	 *
	 * @param int   $contact_id
	 * @param array $data  Contact fields. Optional keys: list_ids (int[]), tag_ids (int[]).
	 * @return bool
	 */
	public static function update( int $contact_id, array $data ): bool {
		$sanitized = self::sanitize_fields( $data );

		// Sync lists if provided.
		if ( isset( $data['list_ids'] ) && is_array( $data['list_ids'] ) ) {
			$new_list_ids = array_map( 'intval', $data['list_ids'] );

			$current_lists   = Contact::get_lists( $contact_id );
			$current_list_ids = array_map( static fn( $l ) => (int) $l['id'], $current_lists );

			// Remove lists no longer in the new set.
			$to_remove = array_diff( $current_list_ids, $new_list_ids );
			foreach ( $to_remove as $list_id ) {
				Contact::unsubscribe_from_list( $contact_id, $list_id );
			}

			// Add lists not yet assigned.
			$to_add = array_diff( $new_list_ids, $current_list_ids );
			foreach ( $to_add as $list_id ) {
				self::subscribe( $contact_id, $list_id );
			}
		}

		// Sync tags if provided.
		if ( isset( $data['tag_ids'] ) && is_array( $data['tag_ids'] ) ) {
			$new_tag_ids = array_map( 'intval', $data['tag_ids'] );

			$current_tags    = Contact::get_tags( $contact_id );
			$current_tag_ids = array_map( static fn( $t ) => (int) $t['id'], $current_tags );

			$to_remove = array_diff( $current_tag_ids, $new_tag_ids );
			foreach ( $to_remove as $tag_id ) {
				Contact::remove_tag( $contact_id, $tag_id );
			}

			$to_add = array_diff( $new_tag_ids, $current_tag_ids );
			foreach ( $to_add as $tag_id ) {
				Contact::add_tag( $contact_id, $tag_id );
			}
		}

		if ( empty( $sanitized ) ) {
			// Nothing else to update in the contacts table.
			return true;
		}

		return Contact::update( $contact_id, $sanitized );
	}

	/**
	 * Delete a contact and all associated pivot records.
	 *
	 * @param int $contact_id
	 * @return bool
	 */
	public static function delete( int $contact_id ): bool {
		return Contact::delete( $contact_id );
	}

	/**
	 * Subscribe a contact to a mailing list.
	 *
	 * Fires the mrme_contact_subscribed action and triggers subscribe automations.
	 *
	 * @param int $contact_id
	 * @param int $list_id
	 * @return void
	 */
	public static function subscribe( int $contact_id, int $list_id ): void {
		Contact::subscribe_to_list( $contact_id, $list_id );

		do_action( 'mrme_contact_subscribed', $contact_id, $list_id );

		AutomationService::trigger_for_contact( $contact_id, 'subscribe' );
	}

	/**
	 * Build the HMAC token string used for unsubscribe links.
	 *
	 * Token payload: contact_id:email:sha1(email . secret_key)
	 *
	 * @param int    $contact_id
	 * @param string $email
	 * @return string  Base64-encoded token.
	 */
	private static function build_unsubscribe_token( int $contact_id, string $email ): string {
		$secret = self::get_secret_key();
		$hash   = sha1( $email . $secret );
		return base64_encode( $contact_id . ':' . $email . ':' . $hash );
	}

	/**
	 * Generate a one-click unsubscribe URL for a contact.
	 *
	 * @param int    $contact_id
	 * @param string $email
	 * @return string
	 */
	public static function generate_unsubscribe_url( int $contact_id, string $email ): string {
		$token = self::build_unsubscribe_token( $contact_id, $email );
		return home_url( '/' ) . '?mrme_action=unsubscribe&token=' . rawurlencode( $token );
	}

	/**
	 * Unsubscribe a contact using a signed token from the URL.
	 *
	 * @param string $token  Base64-encoded token.
	 * @return bool          True on success, false if token is invalid or contact not found.
	 */
	public static function unsubscribe_by_token( string $token ): bool {
		$decoded = base64_decode( $token, true );
		if ( false === $decoded ) {
			return false;
		}

		$parts = explode( ':', $decoded, 3 );
		if ( count( $parts ) !== 3 ) {
			return false;
		}

		[ $contact_id_str, $email, $provided_hash ] = $parts;

		$contact_id = (int) $contact_id_str;
		if ( $contact_id <= 0 ) {
			return false;
		}

		$secret        = self::get_secret_key();
		$expected_hash = sha1( $email . $secret );

		if ( ! hash_equals( $expected_hash, $provided_hash ) ) {
			return false;
		}

		$contact = Contact::find( $contact_id );
		if ( ! $contact ) {
			return false;
		}

		// Verify the email in the token matches the stored email.
		if ( strtolower( $contact['email'] ) !== strtolower( $email ) ) {
			return false;
		}

		return Contact::set_status( $contact_id, 'unsubscribed' );
	}

	/**
	 * Build the confirmation token for double opt-in.
	 *
	 * Token payload: contact_id:email:sha1(email . 'confirm' . secret_key)
	 *
	 * @param int    $contact_id
	 * @param string $email
	 * @return string  Base64-encoded token.
	 */
	private static function build_confirm_token( int $contact_id, string $email ): string {
		$secret = self::get_secret_key();
		$hash   = sha1( $email . 'confirm' . $secret );
		return base64_encode( $contact_id . ':' . $email . ':' . $hash );
	}

	/**
	 * Generate a double opt-in confirmation URL.
	 *
	 * @param int    $contact_id
	 * @param string $email
	 * @return string
	 */
	public static function generate_confirm_url( int $contact_id, string $email ): string {
		$token = self::build_confirm_token( $contact_id, $email );
		return home_url( '/' ) . '?mrme_action=confirm_optin&token=' . rawurlencode( $token );
	}

	/**
	 * Confirm a double opt-in subscription using a signed token.
	 *
	 * Sets status to 'subscribed' and triggers subscribe automations.
	 *
	 * @param string $token  Base64-encoded token.
	 * @return bool
	 */
	public static function confirm_optin_by_token( string $token ): bool {
		$decoded = base64_decode( $token, true );
		if ( false === $decoded ) {
			return false;
		}

		$parts = explode( ':', $decoded, 3 );
		if ( count( $parts ) !== 3 ) {
			return false;
		}

		[ $contact_id_str, $email, $provided_hash ] = $parts;

		$contact_id = (int) $contact_id_str;
		if ( $contact_id <= 0 ) {
			return false;
		}

		$secret        = self::get_secret_key();
		$expected_hash = sha1( $email . 'confirm' . $secret );

		if ( ! hash_equals( $expected_hash, $provided_hash ) ) {
			return false;
		}

		$contact = Contact::find( $contact_id );
		if ( ! $contact ) {
			return false;
		}

		if ( strtolower( $contact['email'] ) !== strtolower( $email ) ) {
			return false;
		}

		$result = Contact::set_status( $contact_id, 'subscribed' );

		if ( $result ) {
			AutomationService::trigger_for_contact( $contact_id, 'subscribe' );
		}

		return $result;
	}

	/**
	 * Import contacts from a CSV file.
	 *
	 * @param string $file_path  Absolute path to the CSV file.
	 * @param array  $options {
	 *     @type int[]  $list_ids        Lists to subscribe imported contacts to.
	 *     @type int[]  $tag_ids         Tags to assign to imported contacts.
	 *     @type bool   $update_existing Whether to update existing contacts (default false).
	 *     @type string $status          Status to assign new contacts (default 'subscribed').
	 * }
	 * @return array { imported: int, skipped: int, errors: string[] }
	 */
	public static function import_csv( string $file_path, array $options = [] ): array {
		$options = wp_parse_args( $options, [
			'list_ids'        => [],
			'tag_ids'         => [],
			'update_existing' => false,
			'status'          => 'subscribed',
		] );

		$result = [
			'imported' => 0,
			'skipped'  => 0,
			'errors'   => [],
		];

		$parsed = CsvImporter::parse( $file_path );

		if ( empty( $parsed['headers'] ) ) {
			$result['errors'][] = __( 'Could not read CSV headers.', 'myrock-mail-engine' );
			return $result;
		}

		$normalized_headers = CsvImporter::normalize_headers( $parsed['headers'] );

		// Ensure there is at least an email column.
		if ( ! in_array( 'email', $normalized_headers, true ) ) {
			$result['errors'][] = __( 'CSV must contain an email column.', 'myrock-mail-engine' );
			return $result;
		}

		foreach ( $parsed['rows'] as $line_number => $row ) {
			$mapped = CsvImporter::map_row( $normalized_headers, $row );

			if ( empty( $mapped['email'] ) || ! is_email( $mapped['email'] ) ) {
				$result['skipped']++;
				$result['errors'][] = sprintf(
					/* translators: %d: CSV line number */
					__( 'Line %d: invalid or missing email address.', 'myrock-mail-engine' ),
					$line_number + 2 // +2: 1-based index + header row.
				);
				continue;
			}

			$email   = sanitize_email( $mapped['email'] );
			$existing = Contact::find_by_email( $email );

			if ( $existing ) {
				if ( ! $options['update_existing'] ) {
					$result['skipped']++;
					continue;
				}

				// Update existing contact.
				$update_data = [];
				if ( ! empty( $mapped['first_name'] ) ) {
					$update_data['first_name'] = sanitize_text_field( $mapped['first_name'] );
				}
				if ( ! empty( $mapped['last_name'] ) ) {
					$update_data['last_name'] = sanitize_text_field( $mapped['last_name'] );
				}
				if ( ! empty( $mapped['phone'] ) ) {
					$update_data['phone'] = sanitize_text_field( $mapped['phone'] );
				}
				if ( ! empty( $mapped['company'] ) ) {
					$update_data['company'] = sanitize_text_field( $mapped['company'] );
				}

				$contact_id = (int) $existing['id'];

				if ( ! empty( $update_data ) ) {
					Contact::update( $contact_id, $update_data );
				}

				// Subscribe / tag regardless.
				if ( ! empty( $options['list_ids'] ) ) {
					foreach ( $options['list_ids'] as $list_id ) {
						self::subscribe( $contact_id, (int) $list_id );
					}
				}
				if ( ! empty( $options['tag_ids'] ) ) {
					foreach ( $options['tag_ids'] as $tag_id ) {
						Contact::add_tag( $contact_id, (int) $tag_id );
					}
				}

				$result['imported']++;
				continue;
			}

			// Build contact data for new record.
			$contact_data = [
				'email'  => $email,
				'status' => sanitize_text_field( $options['status'] ),
			];

			if ( ! empty( $mapped['first_name'] ) ) {
				$contact_data['first_name'] = sanitize_text_field( $mapped['first_name'] );
			}
			if ( ! empty( $mapped['last_name'] ) ) {
				$contact_data['last_name'] = sanitize_text_field( $mapped['last_name'] );
			}
			if ( ! empty( $mapped['phone'] ) ) {
				$contact_data['phone'] = sanitize_text_field( $mapped['phone'] );
			}
			if ( ! empty( $mapped['company'] ) ) {
				$contact_data['company'] = sanitize_text_field( $mapped['company'] );
			}

			$contact_id = Contact::create( $contact_data );

			if ( false === $contact_id ) {
				$result['errors'][] = sprintf(
					/* translators: %s: email address */
					__( 'Failed to create contact for %s.', 'myrock-mail-engine' ),
					$email
				);
				$result['skipped']++;
				continue;
			}

			if ( ! empty( $options['list_ids'] ) ) {
				foreach ( $options['list_ids'] as $list_id ) {
					self::subscribe( $contact_id, (int) $list_id );
				}
			}

			if ( ! empty( $options['tag_ids'] ) ) {
				foreach ( $options['tag_ids'] as $tag_id ) {
					Contact::add_tag( $contact_id, (int) $tag_id );
				}
			}

			$result['imported']++;
		}

		return $result;
	}
}
