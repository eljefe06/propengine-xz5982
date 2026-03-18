<?php

namespace MyRock\MailEngine\Admin\Pages;

defined( 'ABSPATH' ) || exit;

use MyRock\MailEngine\License\LicenseManager;

/**
 * Admin page: License management.
 *
 * @package MyRock\MailEngine\Admin\Pages
 */
class LicensePage {

	/**
	 * Render the license page.
	 *
	 * @return void
	 */
	public function render(): void {
		if ( ! current_user_can( 'manage_options' ) ) {
			wp_die( esc_html__( 'You do not have permission to view this page.', 'myrock-mail-engine' ) );
		}

		$status  = LicenseManager::get_status();
		$key     = LicenseManager::get_key();
		$plan    = LicenseManager::get_plan();
		$is_pro  = LicenseManager::is_pro();

		// phpcs:disable WordPress.Security.NonceVerification.Recommended
		$notice_type = isset( $_GET['mrme_lic_notice'] ) ? sanitize_key( $_GET['mrme_lic_notice'] ) : '';
		// phpcs:enable

		$template = MRME_DIR . 'templates/admin/license.php';
		if ( file_exists( $template ) ) {
			include $template;
		}
	}

	/**
	 * Handle the activate / deactivate form POST.
	 *
	 * @return void
	 */
	public function handle_save(): void {
		if ( ! current_user_can( 'manage_options' ) ) {
			wp_die( esc_html__( 'You do not have permission to perform this action.', 'myrock-mail-engine' ) );
		}

		check_admin_referer( 'mrme_save_license' );

		$redirect = admin_url( 'admin.php?page=mrme-license' );
		$action   = isset( $_POST['license_action'] ) ? sanitize_key( $_POST['license_action'] ) : '';

		if ( 'deactivate' === $action ) {
			LicenseManager::deactivate();
			wp_safe_redirect( add_query_arg( 'mrme_lic_notice', 'deactivated', $redirect ) );
			die();
		}

		// Activate.
		$key    = isset( $_POST['mrme_license_key'] ) ? sanitize_text_field( wp_unslash( $_POST['mrme_license_key'] ) ) : '';
		$result = LicenseManager::activate( $key );

		if ( $result['valid'] ) {
			wp_safe_redirect( add_query_arg( 'mrme_lic_notice', 'activated', $redirect ) );
		} else {
			wp_safe_redirect( add_query_arg( [
				'mrme_lic_notice' => 'invalid',
				'mrme_lic_msg'    => urlencode( $result['message'] ),
			], $redirect ) );
		}
		die();
	}
}
