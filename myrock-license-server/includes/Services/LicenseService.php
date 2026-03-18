<?php

namespace MyRock\LicenseServer\Services;

defined( 'ABSPATH' ) || exit;

use MyRock\LicenseServer\Models\License;

class LicenseService {

	/**
	 * Generate a secure license key.
	 * Format: MRME-XXXX-XXXX-XXXX-XXXX (uppercase alphanum, no ambiguous chars)
	 */
	public static function generate_key(): string {
		$chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
		$key   = 'MRME';
		for ( $g = 0; $g < 4; $g++ ) {
			$key .= '-';
			for ( $i = 0; $i < 4; $i++ ) {
				$key .= $chars[ random_int( 0, strlen( $chars ) - 1 ) ];
			}
		}
		return $key;
	}

	/**
	 * Calculate expires_at based on plan, starting from an optional date.
	 */
	public static function get_expires_at( string $plan, ?string $from = null ): string {
		$dt = $from ? new \DateTime( $from ) : new \DateTime( 'now', wp_timezone() );
		if ( 'annual' === $plan ) {
			$dt->modify( '+1 year' );
		} else {
			$dt->modify( '+1 month' );
		}
		return $dt->format( 'Y-m-d H:i:s' );
	}

	/**
	 * Create a new license after the first successful MP payment.
	 */
	public static function create_from_subscription(
		string $mp_subscription_id,
		string $email,
		string $plan,
		string $mp_payer_email = ''
	): License {
		$license                     = new License();
		$license->license_key        = self::generate_key();
		$license->email              = strtolower( trim( $email ) );
		$license->plan               = in_array( $plan, [ 'monthly', 'annual' ], true ) ? $plan : 'monthly';
		$license->status             = 'active';
		$license->mp_subscription_id = $mp_subscription_id;
		$license->mp_payer_email     = $mp_payer_email ?: $email;
		$license->created_at         = current_time( 'mysql' );
		$license->expires_at         = self::get_expires_at( $license->plan );
		$license->last_renewed_at    = current_time( 'mysql' );
		$license->save();

		self::send_license_email( $license );

		return $license;
	}

	/**
	 * Renew (extend expiry) after a recurring payment.
	 */
	public static function renew( License $license ): void {
		$license->status          = 'active';
		$license->last_renewed_at = current_time( 'mysql' );
		$license->expires_at      = self::get_expires_at( $license->plan );
		$license->save();
	}

	/**
	 * Cancel a license.
	 */
	public static function cancel( License $license ): void {
		$license->status = 'cancelled';
		$license->save();
	}

	/**
	 * Validate a license key. Returns array matching LicenseManager's expected shape.
	 */
	public static function validate( string $key, string $site_url = '' ): array {
		if ( empty( $key ) ) {
			return [ 'valid' => false, 'plan' => 'free', 'message' => 'No key provided.' ];
		}

		$license = License::find_by_key( $key );

		if ( ! $license ) {
			return [ 'valid' => false, 'plan' => 'free', 'message' => 'License key not found.' ];
		}

		if ( ! $license->is_active() ) {
			return [
				'valid'      => false,
				'plan'       => 'free',
				'expires_at' => $license->expires_at,
				'message'    => 'License is ' . $license->status . '.',
			];
		}

		// Track site URL (informational).
		if ( $site_url && $license->site_url !== $site_url ) {
			$license->site_url = esc_url_raw( $site_url );
			$license->save();
		}

		return [
			'valid'      => true,
			'plan'       => 'pro',
			'expires_at' => $license->expires_at,
			'message'    => 'License active.',
		];
	}

	/**
	 * Send the license key to the customer by email.
	 */
	public static function send_license_email( License $license ): void {
		$plan_label = 'annual' === $license->plan ? 'Anual' : 'Mensual';
		$expires    = $license->expires_at
			? wp_date( 'd/m/Y', strtotime( $license->expires_at ) )
			: '—';

		$subject = '[MyRock] Tu licencia Pro de MyRock Mail Engine';

		$body  = "¡Hola!\n\n";
		$body .= "Gracias por adquirir MyRock Mail Engine Pro.\n\n";
		$body .= "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
		$body .= "  Tu llave de licencia:\n\n";
		$body .= "  " . $license->license_key . "\n\n";
		$body .= "  Plan:          Pro $plan_label\n";
		$body .= "  Válida hasta:  $expires\n";
		$body .= "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
		$body .= "Cómo activarla:\n";
		$body .= "1. Ve a WP Admin → MyRock Mail → Licencia\n";
		$body .= "2. Pega la llave y haz clic en «Activar»\n\n";
		$body .= "Tu suscripción se renovará automáticamente y recibirás\n";
		$body .= "un correo de renovación con la misma llave extendida.\n\n";
		$body .= "Dudas: hola@myrock.com.mx\n\n";
		$body .= "— Equipo MyRock\n";
		$body .= "https://myrock.com.mx\n";

		wp_mail( $license->email, $subject, $body );
	}
}
