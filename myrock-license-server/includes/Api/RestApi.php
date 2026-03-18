<?php

namespace MyRock\LicenseServer\Api;

defined( 'ABSPATH' ) || exit;

use MyRock\LicenseServer\Models\License;
use MyRock\LicenseServer\Services\LicenseService;
use MyRock\LicenseServer\Services\MercadoPagoService;

class RestApi {

	public function register_routes(): void {
		// Validate endpoint — called every 24 h by the client plugin's LicenseManager.
		register_rest_route( 'mrme-license/v1', '/validate', [
			'methods'             => \WP_REST_Server::CREATABLE,
			'callback'            => [ $this, 'validate' ],
			'permission_callback' => '__return_true',
		] );

		// MercadoPago webhook (IPN / notifications v2).
		register_rest_route( 'mrls/v1', '/webhook', [
			'methods'             => \WP_REST_Server::CREATABLE,
			'callback'            => [ $this, 'webhook' ],
			'permission_callback' => '__return_true',
		] );

		// Public: return plan pricing + checkout URLs (used by the /plugin page JS).
		register_rest_route( 'mrls/v1', '/plans', [
			'methods'             => \WP_REST_Server::READABLE,
			'callback'            => [ $this, 'get_plans' ],
			'permission_callback' => '__return_true',
		] );

		// Public: create a MercadoPago payment preference and return init_point.
		register_rest_route( 'mrls/v1', '/checkout', [
			'methods'             => \WP_REST_Server::CREATABLE,
			'callback'            => [ $this, 'create_checkout' ],
			'permission_callback' => '__return_true',
			'args'                => [
				'plan'  => [
					'required'          => true,
					'sanitize_callback' => 'sanitize_key',
					'validate_callback' => function( $v ) { return in_array( $v, [ 'monthly', 'annual' ], true ); },
				],
				'email' => [
					'required'          => true,
					'sanitize_callback' => 'sanitize_email',
					'validate_callback' => 'is_email',
				],
			],
		] );
	}

	// -------------------------------------------------------------------------

	public function validate( \WP_REST_Request $request ): \WP_REST_Response {
		$key      = sanitize_text_field( (string) $request->get_param( 'license_key' ) );
		$site_url = esc_url_raw( (string) $request->get_param( 'site_url' ) );
		return new \WP_REST_Response( LicenseService::validate( $key, $site_url ), 200 );
	}

	public function webhook( \WP_REST_Request $request ): \WP_REST_Response {
		$body = json_decode( $request->get_body(), true );

		if ( ! is_array( $body ) ) {
			return new \WP_REST_Response( [ 'ok' => false ], 400 );
		}

		$type    = (string) ( $body['type'] ?? '' );
		$data_id = (string) ( $body['data']['id'] ?? '' );

		error_log( "[MRLS] Webhook: type={$type} id={$data_id}" );

		if ( 'subscription_preapproval' === $type && $data_id ) {
			$this->handle_subscription_event( $data_id );
		} elseif ( in_array( $type, [ 'payment', 'subscription_authorized_payment' ], true ) && $data_id ) {
			$this->handle_payment_event( $data_id );
		}

		return new \WP_REST_Response( [ 'ok' => true ], 200 );
	}

	public function create_checkout( \WP_REST_Request $request ): \WP_REST_Response {
		$plan  = $request->get_param( 'plan' );
		$email = $request->get_param( 'email' );

		$init_point = MercadoPagoService::create_preference( $plan, $email );

		if ( ! $init_point ) {
			return new \WP_REST_Response(
				[ 'error' => 'No se pudo crear el pago. Verifica que el Access Token de MercadoPago esté configurado.' ],
				500
			);
		}

		return new \WP_REST_Response( [ 'init_point' => $init_point ], 200 );
	}

	public function get_plans( \WP_REST_Request $request ): \WP_REST_Response {
		return new \WP_REST_Response( [
			'monthly' => [
				'price'        => (float) get_option( 'mrls_price_monthly', 299 ),
				'currency'     => 'MXN',
				'checkout_url' => (string) get_option( 'mrls_mp_checkout_monthly', '' ),
			],
			'annual' => [
				'price'        => (float) get_option( 'mrls_price_annual', 2499 ),
				'currency'     => 'MXN',
				'checkout_url' => (string) get_option( 'mrls_mp_checkout_annual', '' ),
			],
		], 200 );
	}

	// -------------------------------------------------------------------------
	// Internal handlers
	// -------------------------------------------------------------------------

	private function handle_subscription_event( string $subscription_id ): void {
		$sub = MercadoPagoService::get_subscription( $subscription_id );
		if ( ! $sub ) {
			error_log( "[MRLS] Could not fetch subscription {$subscription_id}" );
			return;
		}

		$mp_status = (string) ( $sub['status'] ?? '' );
		$email     = (string) ( $sub['payer_email'] ?? $sub['payer']['email'] ?? '' );
		$plan      = MercadoPagoService::get_plan_from_subscription( $sub );
		$license   = License::find_by_subscription( $subscription_id );

		if ( 'authorized' === $mp_status ) {
			if ( $license ) {
				LicenseService::renew( $license );
				error_log( "[MRLS] Renewed license {$license->license_key}" );
			} else {
				$new = LicenseService::create_from_subscription( $subscription_id, $email, $plan, $email );
				error_log( "[MRLS] Created license {$new->license_key} for {$email}" );
			}
		} elseif ( in_array( $mp_status, [ 'cancelled', 'paused' ], true ) && $license ) {
			LicenseService::cancel( $license );
			error_log( "[MRLS] Cancelled license {$license->license_key}" );
		}
	}

	private function handle_payment_event( string $payment_id ): void {
		$payment = MercadoPagoService::get_payment( $payment_id );
		if ( ! $payment || 'approved' !== ( $payment['status'] ?? '' ) ) {
			return;
		}

		// Case 1: payment linked to a subscription — renew the license.
		$sub_id = (string) (
			$payment['metadata']['preapproval_id']
			?? $payment['point_of_interaction']['transaction_data']['subscription_id']
			?? ''
		);

		if ( $sub_id ) {
			$license = License::find_by_subscription( $sub_id );
			if ( $license ) {
				LicenseService::renew( $license );
				error_log( "[MRLS] Renewed via payment {$payment_id} → {$license->license_key}" );
			}
			return;
		}

		// Case 2: direct preference payment from the /plugin page checkout modal.
		$ext_ref = (string) ( $payment['external_reference'] ?? '' );
		if ( ! $ext_ref || strpos( $ext_ref, ':' ) === false ) {
			return;
		}

		[ $plan, $email ] = explode( ':', $ext_ref, 2 );
		$plan  = sanitize_key( $plan );
		$email = sanitize_email( $email );

		if ( ! in_array( $plan, [ 'monthly', 'annual' ], true ) || ! is_email( $email ) ) {
			return;
		}

		// Avoid creating a duplicate license for the same payment.
		$existing = License::find_by_subscription( 'pay-' . $payment_id );
		if ( $existing ) {
			return;
		}

		$new = LicenseService::create_from_subscription( 'pay-' . $payment_id, $email, $plan, $email );
		error_log( "[MRLS] Created license {$new->license_key} via payment {$payment_id} for {$email} ({$plan})" );
	}
}
