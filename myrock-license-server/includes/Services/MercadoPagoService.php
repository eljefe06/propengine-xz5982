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

	/**
	 * Create a one-time payment preference in MercadoPago.
	 * Used by the /plugin page checkout modal.
	 *
	 * @param string $plan   'monthly' | 'annual'
	 * @param string $email  Buyer's email.
	 * @return string|null   init_point URL or null on failure.
	 */
	public static function create_preference( string $plan, string $email ): ?string {
		$token = self::access_token();
		if ( ! $token ) {
			return null;
		}

		$price = (float) get_option(
			'annual' === $plan ? 'mrls_price_annual' : 'mrls_price_monthly',
			'annual' === $plan ? 2499 : 299
		);
		$label = 'annual' === $plan ? 'Anual' : 'Mensual';

		$body = wp_json_encode( [
			'items'              => [ [
				'title'       => "MyRock Mail Engine Pro \u2014 {$label}",
				'unit_price'  => $price,
				'quantity'    => 1,
				'currency_id' => 'MXN',
			] ],
			'payer'              => [ 'email' => $email ],
			'external_reference' => "{$plan}:{$email}",
			'back_urls'          => [
				'success' => home_url( '/plugin/?payment=success' ),
				'failure' => home_url( '/plugin/?payment=failure' ),
				'pending' => home_url( '/plugin/?payment=pending' ),
			],
			'notification_url'   => rest_url( 'mrls/v1/webhook' ),
			'auto_return'        => 'approved',
		] );

		$response = wp_remote_post(
			'https://api.mercadopago.com/checkout/preferences',
			[
				'timeout' => 20,
				'headers' => [
					'Authorization' => "Bearer {$token}",
					'Content-Type'  => 'application/json',
				],
				'body' => $body,
			]
		);

		if ( is_wp_error( $response ) ) {
			error_log( '[MRLS] create_preference error: ' . $response->get_error_message() );
			return null;
		}

		$code = wp_remote_retrieve_response_code( $response );
		$data = json_decode( wp_remote_retrieve_body( $response ), true );

		if ( $code < 200 || $code >= 300 || empty( $data['init_point'] ) ) {
			error_log( '[MRLS] create_preference HTTP ' . $code . ': ' . wp_remote_retrieve_body( $response ) );
			return null;
		}

		return (string) $data['init_point'];
	}

	/**
	 * Create both subscription plans (monthly + annual) in MercadoPago.
	 * Returns array with keys 'monthly' and 'annual', each containing the API response.
	 *
	 * @param float  $price_monthly  Amount in MXN for the monthly plan.
	 * @param float  $price_annual   Amount in MXN for the annual plan.
	 * @param string $back_url       URL to redirect after subscription checkout.
	 * @return array{ monthly: array|null, annual: array|null, errors: string[] }
	 */
	public static function create_plans( float $price_monthly, float $price_annual, string $back_url = '' ): array {
		$token = self::access_token();
		if ( ! $token ) {
			return [ 'monthly' => null, 'annual' => null, 'errors' => [ 'Access token not configured.' ] ];
		}

		if ( ! $back_url ) {
			$back_url = home_url( '/plugin/?subscribed=1' );
		}

		$errors  = [];
		$results = [];

		$plans_config = [
			'monthly' => [
				'reason'         => 'MyRock Mail Engine Pro — Mensual',
				'frequency'      => 1,
				'frequency_type' => 'months',
				'amount'         => $price_monthly,
			],
			'annual' => [
				'reason'         => 'MyRock Mail Engine Pro — Anual',
				'frequency'      => 12,
				'frequency_type' => 'months',
				'amount'         => $price_annual,
			],
		];

		foreach ( $plans_config as $key => $cfg ) {
			$body = wp_json_encode( [
				'reason'         => $cfg['reason'],
				'auto_recurring' => [
					'frequency'          => $cfg['frequency'],
					'frequency_type'     => $cfg['frequency_type'],
					'transaction_amount' => $cfg['amount'],
					'currency_id'        => 'MXN',
				],
				'back_url' => $back_url,
				'status'   => 'active',
			] );

			$response = wp_remote_post(
				'https://api.mercadopago.com/preapproval_plan',
				[
					'timeout' => 20,
					'headers' => [
						'Authorization' => "Bearer {$token}",
						'Content-Type'  => 'application/json',
					],
					'body' => $body,
				]
			);

			if ( is_wp_error( $response ) ) {
				$errors[]       = "{$key}: " . $response->get_error_message();
				$results[ $key ] = null;
				continue;
			}

			$code = wp_remote_retrieve_response_code( $response );
			$data = json_decode( wp_remote_retrieve_body( $response ), true );

			if ( $code < 200 || $code >= 300 || empty( $data['id'] ) ) {
				$msg            = $data['message'] ?? "HTTP {$code}";
				$errors[]       = "{$key}: {$msg}";
				$results[ $key ] = null;
				continue;
			}

			$results[ $key ] = $data;
		}

		$results['errors'] = $errors;
		return $results;
	}
}
