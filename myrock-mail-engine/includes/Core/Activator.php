<?php
/**
 * Plugin activation, deactivation and uninstall.
 *
 * @package MyRock\MailEngine\Core
 */

namespace MyRock\MailEngine\Core;

defined( 'ABSPATH' ) || exit;

class Activator {

	/**
	 * Run on plugin activation.
	 * - Creates/upgrades DB tables
	 * - Stores plugin version
	 * - Schedules cron
	 */
	public static function activate(): void {
		\MyRock\MailEngine\Database\Schema::create_tables();
		update_option( 'mrme_version',    MRME_VERSION );
		update_option( 'mrme_db_version', MRME_DB_VERSION );
		\MyRock\MailEngine\Database\Schema::seed_defaults();
		flush_rewrite_rules();
	}

	public static function uninstall(): void {
		// Only remove data if the setting says so
		if ( get_option( 'mrme_delete_on_uninstall' ) !== '1' ) return;

		global $wpdb;
		$tables = [
			'mrme_contacts', 'mrme_lists', 'mrme_contact_lists',
			'mrme_tags',     'mrme_contact_tags',
			'mrme_campaigns','mrme_send_logs', 'mrme_events',
			'mrme_forms',    'mrme_automations', 'mrme_automation_steps',
			'mrme_automation_queue',
		];
		foreach ( $tables as $table ) {
			$wpdb->query( "DROP TABLE IF EXISTS `{$wpdb->prefix}{$table}`" ); // phpcs:ignore
		}

		// Remove options
		$options = [
			'mrme_version', 'mrme_db_version', 'mrme_settings',
			'mrme_delete_on_uninstall',
		];
		foreach ( $options as $opt ) {
			delete_option( $opt );
		}
	}
}
