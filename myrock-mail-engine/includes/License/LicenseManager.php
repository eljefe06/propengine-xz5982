<?php

namespace MyRock\MailEngine\License;

defined( 'ABSPATH' ) || exit;

/**
 * Class LicenseManager
 *
 * Manages plugin license validation and feature gating.
 *
 * Plans:
 *   - free  : up to 500 contacts, 1 active campaign at a time, no Mailgun, no automations.
 *   - pro   : unlimited contacts, all providers (including Mailgun), automations, priority support.
 *
 * License keys are validated against the MyRock licensing API.
 * The result is cached in a transient for 24 hours to avoid hammering the API.
 *
 * Option key : mrme_license
 * Transient  : mrme_license_status
 *
 * @package MyRock\MailEngine\License
 */
class LicenseManager {

	/** WordPress option that stores the raw license key. */
	const OPTION_KEY = 'mrme_license';

	/** Transient that caches the validated license status. */
	const TRANSIENT_KEY = 'mrme_license_status';

	/** How long to cache a successful validation (seconds). */
	const CACHE_TTL = DAY_IN_SECONDS;

	/** Licensing API endpoint. Change this to your own server. */
	const API_URL = 'https://myrock.com.mx/wp-json/mrme-license/v1/validate';

	/** Plans. */
	const PLAN_FREE = 'free';
	const PLAN_PRO  = 'pro';

	/** Contact limit for the free plan. */
	const FREE_CONTACT_LIMIT = 500;

	/** Campaign limit for the free plan. */
	const FREE_CAMPAIGN_LIMIT = 1;

	// -------------------------------------------------------------------------
	// Public API
	// -------------------------------------------------------------------------

	/**
	 * Return the current plan: 'free' or 'pro'.
	 *
	 * @return string
	 */
	public static function get_plan(): string {
		$status = self::get_status();
		return $status['plan'] ?? self::PLAN_FREE;
	}

	/**
	 * Is the current install running the Pro plan?
	 *
	 * @return bool
	 */
	public static function is_pro(): bool {
		return self::PLAN_PRO === self::get_plan();
	}

	/**
	 * Return the stored license key (raw, un-validated).
	 *
	 * @return string
	 */
	public static function get_key(): string {
		$data = get_option( self::OPTION_KEY, [] );
		return $data['key'] ?? '';
	}

	/**
	 * Return the cached license status array.
	 *
	 * Shape: [ 'valid' => bool, 'plan' => string, 'expires' => string|null, 'message' => string ]
	 *
	 * @return array
	 */
	public static function get_status(): array {
		$cached = get_transient( self::TRANSIENT_KEY );
		if ( is_array( $cached ) ) {
			return $cached;
		}

		$key = self::get_key();
		if ( empty( $key ) ) {
			return self::free_status( __( 'No license key entered.', 'myrock-mail-engine' ) );
		}

		return self::validate_key( $key );
	}

	/**
	 * Activate a license key.
	 *
	 * Stores the key and forces a fresh API validation.
	 *
	 * @param string $key Raw license key submitted by the user.
	 * @return array Validation result.
	 */
	public static function activate( string $key ): array {
		$key = sanitize_text_field( trim( $key ) );

		if ( empty( $key ) ) {
			return self::free_status( __( 'Please enter a license key.', 'myrock-mail-engine' ) );
		}

		// Persist key first so validate_key can send it.
		update_option( self::OPTION_KEY, [ 'key' => $key ], false );
		delete_transient( self::TRANSIENT_KEY );

		$result = self::validate_key( $key );

		return $result;
	}

	/**
	 * Deactivate the current license (clears key and cache).
	 *
	 * @return void
	 */
	public static function deactivate(): void {
		delete_option( self::OPTION_KEY );
		delete_transient( self::TRANSIENT_KEY );
	}

	// -------------------------------------------------------------------------
	// Feature gates
	// -------------------------------------------------------------------------

	/**
	 * Can the current plan use Mailgun?
	 *
	 * @return bool
	 */
	public static function can_use_mailgun(): bool {
		return self::is_pro();
	}

	/**
	 * Can the current plan use Automations?
	 *
	 * @return bool
	 */
	public static function can_use_automations(): bool {
		return self::is_pro();
	}

	/**
	 * Can the current plan add more contacts?
	 *
	 * @param int $current_count Current number of contacts in the DB.
	 * @return bool
	 */
	public static function can_add_contact( int $current_count ): bool {
		if ( self::is_pro() ) {
			return true;
		}
		return $current_count < self::FREE_CONTACT_LIMIT;
	}

	/**
	 * Return human-readable plan label.
	 *
	 * @return string
	 */
	public static function get_plan_label(): string {
		return self::is_pro()
			? __( 'Pro', 'myrock-mail-engine' )
			: __( 'Free', 'myrock-mail-engine' );
	}

	// -------------------------------------------------------------------------
	// Internal helpers
	// -------------------------------------------------------------------------

	/**
	 * Call the licensing API and return a normalised status array.
	 *
	 * Falls back to free plan on any network error so the plugin remains usable.
	 *
	 * @param string $key License key to validate.
	 * @return array
	 */
	private static function validate_key( string $key ): array {
		$response = wp_remote_post( self::API_URL, [
			'timeout' => 15,
			'body'    => [
				'license_key' => $key,
				'site_url'    => home_url(),
				'plugin'      => 'myrock-mail-engine',
			],
		] );

		// Network error → allow free plan so the plugin doesn't break.
		if ( is_wp_error( $response ) ) {
			$status = self::free_status(
				sprintf(
					/* translators: %s error message */
					__( 'Could not reach the licensing server: %s', 'myrock-mail-engine' ),
					$response->get_error_message()
				)
			);
			set_transient( self::TRANSIENT_KEY, $status, HOUR_IN_SECONDS );
			return $status;
		}

		$code = wp_remote_retrieve_response_code( $response );
		$body = json_decode( wp_remote_retrieve_body( $response ), true );

		if ( 200 === $code && isset( $body['valid'] ) && true === $body['valid'] ) {
			$plan    = ( isset( $body['plan'] ) && 'pro' === strtolower( $body['plan'] ) )
				? self::PLAN_PRO
				: self::PLAN_FREE;
			$expires = $body['expires_at'] ?? null;
			$status  = [
				'valid'   => true,
				'plan'    => $plan,
				'expires' => $expires,
				'message' => __( 'License active.', 'myrock-mail-engine' ),
			];
			set_transient( self::TRANSIENT_KEY, $status, self::CACHE_TTL );
			return $status;
		}

		$msg = $body['message'] ?? __( 'Invalid or expired license key.', 'myrock-mail-engine' );
		$status = self::free_status( $msg );
		set_transient( self::TRANSIENT_KEY, $status, HOUR_IN_SECONDS );
		return $status;
	}

	/**
	 * Return a normalised "free / invalid" status array.
	 *
	 * @param string $message Human-readable explanation.
	 * @return array
	 */
	private static function free_status( string $message ): array {
		return [
			'valid'   => false,
			'plan'    => self::PLAN_FREE,
			'expires' => null,
			'message' => $message,
		];
	}
}
