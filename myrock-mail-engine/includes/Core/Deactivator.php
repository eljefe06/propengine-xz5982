<?php
/**
 * Plugin deactivation — cleans cron, flushes rewrite rules.
 *
 * @package MyRock\MailEngine\Core
 */

namespace MyRock\MailEngine\Core;

defined( 'ABSPATH' ) || exit;

class Deactivator {

	public static function deactivate(): void {
		wp_clear_scheduled_hook( 'mrme_process_campaigns' );
		wp_clear_scheduled_hook( 'mrme_process_automations' );
		flush_rewrite_rules();
	}
}
