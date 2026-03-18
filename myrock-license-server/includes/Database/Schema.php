<?php

namespace MyRock\LicenseServer\Database;

defined( 'ABSPATH' ) || exit;

class Schema {

	public static function create_tables(): void {
		global $wpdb;

		$charset = $wpdb->get_charset_collate();
		$table   = $wpdb->prefix . 'mrls_licenses';

		$sql = "CREATE TABLE {$table} (
			id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			license_key VARCHAR(64) NOT NULL,
			email VARCHAR(191) NOT NULL,
			plan VARCHAR(20) NOT NULL DEFAULT 'monthly',
			status VARCHAR(20) NOT NULL DEFAULT 'pending',
			mp_subscription_id VARCHAR(100) DEFAULT NULL,
			mp_payer_email VARCHAR(191) DEFAULT NULL,
			site_url VARCHAR(500) DEFAULT NULL,
			created_at DATETIME NOT NULL,
			expires_at DATETIME DEFAULT NULL,
			last_renewed_at DATETIME DEFAULT NULL,
			notes TEXT DEFAULT NULL,
			PRIMARY KEY  (id),
			UNIQUE KEY license_key (license_key),
			KEY email (email),
			KEY mp_subscription_id (mp_subscription_id),
			KEY status (status)
		) {$charset};";

		require_once ABSPATH . 'wp-admin/includes/upgrade.php';
		dbDelta( $sql );

		update_option( 'mrls_db_version', MRLS_VERSION );
	}
}
