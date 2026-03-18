<?php

namespace MyRock\LicenseServer\Admin;

defined( 'ABSPATH' ) || exit;

use MyRock\LicenseServer\Models\License;
use MyRock\LicenseServer\Services\LicenseService;

class Admin {

	public function init(): void {
		add_action( 'admin_menu', [ $this, 'register_menus' ] );
		add_action( 'admin_post_mrls_save_settings',    [ $this, 'handle_settings' ] );
		add_action( 'admin_post_mrls_create_license',   [ $this, 'handle_create_license' ] );
		add_action( 'admin_post_mrls_cancel_license',   [ $this, 'handle_cancel_license' ] );
	}

	public function register_menus(): void {
		add_menu_page(
			__( 'Licencias MyRock', 'mrls' ),
			__( 'Licencias MyRock', 'mrls' ),
			'manage_options',
			'mrls-licenses',
			[ $this, 'render_licenses' ],
			'dashicons-admin-network',
			82
		);
		add_submenu_page(
			'mrls-licenses',
			__( 'Configuración', 'mrls' ),
			__( 'Configuración', 'mrls' ),
			'manage_options',
			'mrls-settings',
			[ $this, 'render_settings' ]
		);
	}

	public function render_licenses(): void {
		if ( ! current_user_can( 'manage_options' ) ) {
			wp_die( 'No autorizado.' );
		}
		include MRLS_DIR . 'templates/admin/licenses.php';
	}

	public function render_settings(): void {
		if ( ! current_user_can( 'manage_options' ) ) {
			wp_die( 'No autorizado.' );
		}
		include MRLS_DIR . 'templates/admin/settings.php';
	}

	public function handle_settings(): void {
		if ( ! current_user_can( 'manage_options' ) ) {
			wp_die();
		}
		check_admin_referer( 'mrls_save_settings' );

		$fields = [
			'mrls_mp_access_token'    => 'sanitize_text_field',
			'mrls_mp_plan_monthly_id' => 'sanitize_text_field',
			'mrls_mp_plan_annual_id'  => 'sanitize_text_field',
			'mrls_mp_checkout_monthly' => 'esc_url_raw',
			'mrls_mp_checkout_annual'  => 'esc_url_raw',
			'mrls_price_monthly'      => 'floatval',
			'mrls_price_annual'       => 'floatval',
		];

		foreach ( $fields as $key => $sanitizer ) {
			if ( isset( $_POST[ $key ] ) ) {
				// phpcs:ignore WordPress.Security.ValidatedSanitizedInput.InputNotSanitized
				update_option( $key, $sanitizer( wp_unslash( $_POST[ $key ] ) ) );
			}
		}

		wp_safe_redirect( add_query_arg( 'updated', '1', admin_url( 'admin.php?page=mrls-settings' ) ) );
		die();
	}

	public function handle_create_license(): void {
		if ( ! current_user_can( 'manage_options' ) ) {
			wp_die();
		}
		check_admin_referer( 'mrls_create_license' );

		$email = sanitize_email( wp_unslash( $_POST['email'] ?? '' ) );
		$plan  = sanitize_key( $_POST['plan'] ?? 'monthly' );
		$notes = sanitize_textarea_field( wp_unslash( $_POST['notes'] ?? '' ) );

		if ( $email ) {
			$license        = LicenseService::create_from_subscription( 'manual-' . uniqid(), $email, $plan );
			$license->notes = $notes;
			$license->save();
		}

		wp_safe_redirect( add_query_arg( 'created', '1', admin_url( 'admin.php?page=mrls-licenses' ) ) );
		die();
	}

	public function handle_cancel_license(): void {
		if ( ! current_user_can( 'manage_options' ) ) {
			wp_die();
		}
		check_admin_referer( 'mrls_cancel_license' );

		$id      = (int) ( $_POST['license_id'] ?? 0 );
		$license = License::find( $id );
		if ( $license ) {
			LicenseService::cancel( $license );
		}

		wp_safe_redirect( admin_url( 'admin.php?page=mrls-licenses&cancelled=1' ) );
		die();
	}
}
