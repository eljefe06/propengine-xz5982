<?php
/**
 * Main plugin bootstrap — singleton.
 *
 * @package MyRock\MailEngine\Core
 */

namespace MyRock\MailEngine\Core;

defined( 'ABSPATH' ) || exit;

final class Plugin {

	/** @var self|null */
	private static ?self $instance = null;

	private Loader $loader;

	private function __construct() {
		$this->loader = new Loader();
	}

	public static function instance(): self {
		if ( null === self::$instance ) {
			self::$instance = new self();
		}
		return self::$instance;
	}

	/**
	 * Boot all plugin subsystems.
	 */
	public function run(): void {
		$this->load_textdomain();
		$this->register_admin();
		$this->register_public();
		$this->register_api();
		$this->register_cron();
		$this->register_integrations();
		$this->loader->run();
	}

	private function load_textdomain(): void {
		// Override locale with the plugin's own language setting.
		$settings    = get_option( 'mrme_settings', [] );
		$plugin_lang = $settings['plugin_lang'] ?? '';

		if ( ! empty( $plugin_lang ) && 'en' !== $plugin_lang ) {
			$mo_file = MRME_DIR . 'languages/myrock-mail-engine-' . $plugin_lang . '.mo';
			if ( file_exists( $mo_file ) ) {
				load_textdomain( 'myrock-mail-engine', $mo_file );
				return;
			}
		}

		load_plugin_textdomain(
			'myrock-mail-engine',
			false,
			dirname( MRME_BASENAME ) . '/languages/'
		);
	}

	private function register_admin(): void {
		if ( ! is_admin() ) return;

		$admin = new \MyRock\MailEngine\Admin\Admin();
		$this->loader->add_action( 'admin_menu',            $admin, 'register_menus' );
		$this->loader->add_action( 'admin_enqueue_scripts', $admin, 'enqueue_assets' );
		$this->loader->add_action( 'admin_post_mrme_save_contact',  $admin, 'handle_save_contact' );
		$this->loader->add_action( 'admin_post_mrme_delete_contact',$admin, 'handle_delete_contact' );
		$this->loader->add_action( 'admin_post_mrme_import_csv',    $admin, 'handle_import_csv' );
		$this->loader->add_action( 'admin_post_mrme_save_list',     $admin, 'handle_save_list' );
		$this->loader->add_action( 'admin_post_mrme_save_tag',      $admin, 'handle_save_tag' );
		$this->loader->add_action( 'admin_post_mrme_save_form',     $admin, 'handle_save_form' );
		$this->loader->add_action( 'admin_post_mrme_save_campaign', $admin, 'handle_save_campaign' );
		$this->loader->add_action( 'admin_post_mrme_send_campaign', $admin, 'handle_send_campaign' );
		$this->loader->add_action( 'admin_post_mrme_save_settings', $admin, 'handle_save_settings' );
		$this->loader->add_action( 'admin_post_mrme_save_license',  $admin, 'handle_save_license' );
		$this->loader->add_action( 'wp_ajax_mrme_send_test',        $admin, 'ajax_send_test' );
		$this->loader->add_action( 'wp_ajax_mrme_test_mailgun',     $admin, 'ajax_test_mailgun' );
		$this->loader->add_action( 'wp_ajax_mrme_search_contacts',  $admin, 'ajax_search_contacts' );
	}

	private function register_public(): void {
		$public = new \MyRock\MailEngine\Public\Shortcodes();
		$this->loader->add_action( 'init',                $public, 'register_shortcodes' );
		$this->loader->add_action( 'wp_enqueue_scripts',  $public, 'enqueue_assets' );
		$this->loader->add_action( 'admin_post_nopriv_mrme_subscribe', $public, 'handle_subscribe' );
		$this->loader->add_action( 'admin_post_mrme_subscribe',        $public, 'handle_subscribe' );

		// Unsubscribe page
		$handler = new \MyRock\MailEngine\Public\FormHandler();
		$this->loader->add_action( 'init', $handler, 'handle_unsubscribe' );
		$this->loader->add_action( 'init', $handler, 'handle_confirm_optin' );
	}

	private function register_api(): void {
		$api = new \MyRock\MailEngine\Api\RestApi();
		$this->loader->add_action( 'rest_api_init', $api, 'register_routes' );
	}

	private function register_cron(): void {
		// Schedule campaign sending
		if ( ! wp_next_scheduled( 'mrme_process_campaigns' ) ) {
			wp_schedule_event( time(), 'every_five_minutes', 'mrme_process_campaigns' );
		}
		add_action( 'mrme_process_campaigns', [ \MyRock\MailEngine\Services\CampaignService::class, 'process_scheduled' ] );

		// Schedule automation steps
		if ( ! wp_next_scheduled( 'mrme_process_automations' ) ) {
			wp_schedule_event( time(), 'every_five_minutes', 'mrme_process_automations' );
		}
		add_action( 'mrme_process_automations', [ \MyRock\MailEngine\Services\AutomationService::class, 'process_pending_steps' ] );

		// Register custom interval
		add_filter( 'cron_schedules', function( $schedules ) {
			$schedules['every_five_minutes'] = [
				'interval' => 300,
				'display'  => 'Every 5 minutes',
			];
			return $schedules;
		} );
	}

	private function register_integrations(): void {
		// WPForms → MRME: captura leads aunque sea con la versión Lite.
		// Solo se activa si WPForms está instalado.
		if ( defined( 'WPFORMS_VERSION' ) ) {
			$wpforms = new \MyRock\MailEngine\Integrations\WpFormsIntegration();
			$this->loader->add_action( 'wpforms_process_complete', $wpforms, 'handle_submission', 10, 4 );
		}
	}

	public function get_loader(): Loader {
		return $this->loader;
	}
}
