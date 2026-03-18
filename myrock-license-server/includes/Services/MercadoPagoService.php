<?php

namespace MyRock\LicenseServer\Services;

defined( 'ABSPATH' ) || exit;

class MercadoPagoService {

	private static function access_token(): string {
		return (string) get_option( 'mrls_mp_access_token', '' );
	}

	/**
	 * Fetch subscription (preapproval) details from MP API.
	 */
	public static function get_subscription( string $subscription_id ): ?array {
		$token = self::access_token();
		if ( ! $token ) {
			return null;
		}

		$response = wp_remote_get(
			"https://api.mercadopago.com/preapproval/{$subscription_id}",
			[
				'timeout' => 15,
				'headers' => [ 'Authorization' => "Bearer {$token}" ],
			]
		);

		if ( is_wp_error( $response ) || 200 !== wp_remote_retrieve_response_code( $response ) ) {
			return null;
		}

		return json_decode( wp_remote_retrieve_body( $response ), true );
	}

	/**
	 * Fetch an individual payment from MP API.
	 */
	public static function get_payment( string $payment_id ): ?array {
		$token = self::access_token();
		if ( ! $token ) {
			return null;
		}

		$response = wp_remote_get(
			"https://api.mercadopago.com/v1/payments/{$payment_id}",
			[
				'timeout' => 15,
				'headers' => [ 'Authorization' => "Bearer {$token}" ],
			]
		);

		if ( is_wp_error( $response ) || 200 !== wp_remote_retrieve_response_code( $response ) ) {
			return null;
		}

		return json_decode( wp_remote_retrieve_body( $response ), true );
	}

	/**
	 * Determine plan from subscription by comparing its preapproval_plan_id
	 * against the configured plan IDs in settings.
	 */
	public static function get_plan_from_subscription( array $subscription ): string {
		$annual_id = (string) get_option( 'mrls_mp_plan_annual_id', '' );
		$plan_id   = (string) ( $subscription['preapproval_plan_id'] ?? '' );

		return ( $annual_id && $plan_id === $annual_id ) ? 'annual' : 'monthly';
	}
}
