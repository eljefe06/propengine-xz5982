<?php
namespace MyRock\MailEngine\Admin;

defined('ABSPATH') || exit;


use MyRock\MailEngine\Admin\Pages\ContactsPage;
use MyRock\MailEngine\Admin\Pages\ListsPage;
use MyRock\MailEngine\Admin\Pages\TagsPage;
use MyRock\MailEngine\Admin\Pages\FormsPage;
use MyRock\MailEngine\Admin\Pages\CampaignsPage;
use MyRock\MailEngine\Admin\Pages\SettingsPage;
use MyRock\MailEngine\Admin\Pages\DashboardPage;
use MyRock\MailEngine\Admin\Pages\AutomationsPage;
use MyRock\MailEngine\Admin\Pages\LogsPage;
use MyRock\MailEngine\Admin\Pages\LicensePage;
use MyRock\MailEngine\Services\CampaignService;

/**
 * Main Admin class for MyRock Mail Engine.
 * Registered via Plugin::register_admin().
 */
class Admin {

    /**
     * Register WordPress admin menus and submenus.
     *
     * @return void
     */
    public function register_menus(): void {
        add_menu_page(
            __('MyRock Mail', 'myrock-mail-engine'),
            __('MyRock Mail', 'myrock-mail-engine'),
            'manage_options',
            'mrme-dashboard',
            '',
            'dashicons-email-alt',
            25
        );

        add_submenu_page(
            'mrme-dashboard',
            __('Dashboard', 'myrock-mail-engine'),
            __('Dashboard', 'myrock-mail-engine'),
            'manage_options',
            'mrme-dashboard',
            [ new DashboardPage(), 'render' ]
        );

        add_submenu_page(
            'mrme-dashboard',
            __('Contacts', 'myrock-mail-engine'),
            __('Contacts', 'myrock-mail-engine'),
            'manage_options',
            'mrme-contacts',
            [ new ContactsPage(), 'render' ]
        );

        add_submenu_page(
            'mrme-dashboard',
            __('Lists', 'myrock-mail-engine'),
            __('Lists', 'myrock-mail-engine'),
            'manage_options',
            'mrme-lists',
            [ new ListsPage(), 'render' ]
        );

        add_submenu_page(
            'mrme-dashboard',
            __('Tags', 'myrock-mail-engine'),
            __('Tags', 'myrock-mail-engine'),
            'manage_options',
            'mrme-tags',
            [ new TagsPage(), 'render' ]
        );

        add_submenu_page(
            'mrme-dashboard',
            __('Forms', 'myrock-mail-engine'),
            __('Forms', 'myrock-mail-engine'),
            'manage_options',
            'mrme-forms',
            [ new FormsPage(), 'render' ]
        );

        add_submenu_page(
            'mrme-dashboard',
            __('Campaigns', 'myrock-mail-engine'),
            __('Campaigns', 'myrock-mail-engine'),
            'manage_options',
            'mrme-campaigns',
            [ new CampaignsPage(), 'render' ]
        );

        add_submenu_page(
            'mrme-dashboard',
            __('Automations', 'myrock-mail-engine'),
            __('Automations', 'myrock-mail-engine'),
            'manage_options',
            'mrme-automations',
            [ new AutomationsPage(), 'render' ]
        );

        add_submenu_page(
            'mrme-dashboard',
            __('Logs', 'myrock-mail-engine'),
            __('Logs', 'myrock-mail-engine'),
            'manage_options',
            'mrme-logs',
            [ new LogsPage(), 'render' ]
        );

        add_submenu_page(
            'mrme-dashboard',
            __('Settings', 'myrock-mail-engine'),
            __('Settings', 'myrock-mail-engine'),
            'manage_options',
            'mrme-settings',
            [ new SettingsPage(), 'render' ]
        );

        add_submenu_page(
            'mrme-dashboard',
            __('License', 'myrock-mail-engine'),
            __('License', 'myrock-mail-engine') . $this->license_badge(),
            'manage_options',
            'mrme-license',
            [ new LicensePage(), 'render' ]
        );
    }

    /**
     * Enqueue admin assets only on MRME admin pages.
     *
     * @param string $hook Current admin page hook suffix.
     * @return void
     */
    public function enqueue_assets( string $hook ): void {
        // Only load on mrme-* pages. The hook for top-level menus looks like
        // "toplevel_page_mrme-dashboard" and for submenus like
        // "myrock-mail_page_mrme-contacts", so we check for the slug pattern.
        if (
            strpos( $hook, 'mrme-' ) === false
            && strpos( $hook, 'toplevel_page_mrme' ) === false
        ) {
            return;
        }

        wp_enqueue_style(
            'mrme-admin-css',
            MRME_URL . 'assets/css/admin.css',
            [],
            MRME_VERSION
        );

        wp_enqueue_script(
            'mrme-admin-js',
            MRME_URL . 'assets/js/admin.js',
            [ 'jquery', 'wp-editor' ],
            MRME_VERSION,
            true
        );

        wp_localize_script(
            'mrme-admin-js',
            'mrme',
            [
                'ajax_url' => admin_url( 'admin-ajax.php' ),
                'nonce'    => wp_create_nonce( 'mrme_ajax' ),
            ]
        );
    }

    /**
     * Delegate contact save to ContactsPage.
     *
     * @return void
     */
    public function handle_save_contact(): void {
        ( new ContactsPage() )->handle_save();
    }

    /**
     * Delegate contact delete to ContactsPage.
     *
     * @return void
     */
    public function handle_delete_contact(): void {
        ( new ContactsPage() )->handle_delete();
    }

    /**
     * Delegate CSV import to ContactsPage.
     *
     * @return void
     */
    public function handle_import_csv(): void {
        ( new ContactsPage() )->handle_import();
    }

    /**
     * Delegate list save to ListsPage.
     *
     * @return void
     */
    public function handle_save_list(): void {
        ( new ListsPage() )->handle_save();
    }

    /**
     * Delegate tag save to TagsPage.
     *
     * @return void
     */
    public function handle_save_tag(): void {
        ( new TagsPage() )->handle_save();
    }

    /**
     * Delegate form save to FormsPage.
     *
     * @return void
     */
    public function handle_save_form(): void {
        ( new FormsPage() )->handle_save();
    }

    /**
     * Delegate campaign save to CampaignsPage.
     *
     * @return void
     */
    public function handle_save_campaign(): void {
        ( new CampaignsPage() )->handle_save();
    }

    /**
     * Delegate campaign send to CampaignsPage.
     *
     * @return void
     */
    public function handle_send_campaign(): void {
        ( new CampaignsPage() )->handle_send();
    }

    /**
     * Delegate settings save to SettingsPage.
     *
     * @return void
     */
    public function handle_save_settings(): void {
        ( new SettingsPage() )->handle_save();
    }

    public function handle_save_license(): void {
        ( new LicensePage() )->handle_save();
    }

    /**
     * Return a small badge to append to the License menu item when on free plan.
     *
     * @return string
     */
    private function license_badge(): string {
        if ( \MyRock\MailEngine\License\LicenseManager::is_pro() ) {
            return '';
        }
        return ' <span style="background:#f0ad4e;color:#fff;border-radius:3px;padding:1px 6px;font-size:10px;vertical-align:middle;font-weight:700;">FREE</span>';
    }

    /**
     * AJAX handler: send a test email for a campaign.
     *
     * @return void
     */
    public function ajax_send_test(): void {
        check_ajax_referer( 'mrme_ajax', 'nonce' );

        if ( ! current_user_can( 'manage_options' ) ) {
            wp_send_json_error( [ 'message' => __( 'Insufficient permissions.', 'myrock-mail-engine' ) ] );
        }

        $campaign_id = isset( $_POST['campaign_id'] ) ? (int) $_POST['campaign_id'] : 0;
        $to_email    = isset( $_POST['email'] ) ? sanitize_email( wp_unslash( $_POST['email'] ) ) : '';

        if ( ! $campaign_id || ! is_email( $to_email ) ) {
            wp_send_json_error( [ 'message' => __( 'Invalid campaign ID or email address.', 'myrock-mail-engine' ) ] );
        }

        $service = new CampaignService();
        $result  = $service->send_test( $campaign_id, $to_email );

        if ( is_wp_error( $result ) ) {
            wp_send_json_error( [ 'message' => $result->get_error_message() ] );
        }

        wp_send_json_success( [ 'message' => __( 'Test email sent successfully.', 'myrock-mail-engine' ) ] );
    }

    /**
     * AJAX handler: search contacts by query string.
     *
     * @return void
     */
    public function ajax_search_contacts(): void {
        check_ajax_referer( 'mrme_ajax', 'nonce' );

        if ( ! current_user_can( 'manage_options' ) ) {
            wp_send_json_error( [ 'message' => __( 'Insufficient permissions.', 'myrock-mail-engine' ) ] );
        }

        global $wpdb;

        $query   = isset( $_GET['q'] ) ? sanitize_text_field( wp_unslash( $_GET['q'] ) ) : '';
        $query   = '%' . $wpdb->esc_like( $query ) . '%';
        $table   = $wpdb->prefix . 'mrme_contacts';

        // phpcs:ignore WordPress.DB.DirectDatabaseQuery, WordPress.DB.PreparedSQL.InterpolatedNotPrepared
        $results = $wpdb->get_results(
            $wpdb->prepare(
                "SELECT id, email, first_name, last_name FROM {$table}
                 WHERE email LIKE %s OR first_name LIKE %s OR last_name LIKE %s
                 ORDER BY email ASC
                 LIMIT 20",
                $query,
                $query,
                $query
            ),
            ARRAY_A
        );

        if ( null === $results ) {
            $results = [];
        }

        $contacts = array_map( function ( $row ) {
            return [
                'id'         => (int) $row['id'],
                'email'      => $row['email'],
                'first_name' => $row['first_name'],
                'last_name'  => $row['last_name'],
            ];
        }, $results );

        wp_send_json_success( $contacts );
    }

    /**
     * AJAX handler: send a test email via Mailgun with current form values.
     *
     * @return void
     */
    public function ajax_test_mailgun(): void {
        check_ajax_referer( 'mrme_test_mailgun', '_wpnonce' );

        if ( ! current_user_can( 'manage_options' ) ) {
            wp_send_json_error( [ 'message' => __( 'Insufficient permissions.', 'myrock-mail-engine' ) ] );
        }

        $api_key = isset( $_POST['api_key'] ) ? sanitize_text_field( wp_unslash( $_POST['api_key'] ) ) : '';
        $domain  = isset( $_POST['domain'] )  ? sanitize_text_field( wp_unslash( $_POST['domain'] ) )  : '';
        $region  = isset( $_POST['region'] )  ? sanitize_key( $_POST['region'] ) : 'us';
        $email   = isset( $_POST['email'] )   ? sanitize_email( wp_unslash( $_POST['email'] ) ) : '';

        // Fall back to stored key if the field was left blank (already saved).
        if ( empty( $api_key ) ) {
            $s       = get_option( 'mrme_settings', [] );
            $api_key = $s['mailgun_api_key'] ?? '';
        }

        if ( empty( $api_key ) || empty( $domain ) || empty( $email ) ) {
            wp_send_json_error( [ 'message' => __( 'API key, domain and a From email are required.', 'myrock-mail-engine' ) ] );
        }

        $base_url = ( 'eu' === $region )
            ? 'https://api.eu.mailgun.net/v3'
            : 'https://api.mailgun.net/v3';

        $response = wp_remote_post( $base_url . '/' . $domain . '/messages', [
            'timeout' => 20,
            'headers' => [
                'Authorization' => 'Basic ' . base64_encode( 'api:' . $api_key ),
            ],
            'body' => [
                'from'    => get_bloginfo( 'name' ) . ' <' . $email . '>',
                'to'      => $email,
                'subject' => '[MyRock Mail Engine] Mailgun test',
                'text'    => __( 'Mailgun is configured correctly. This is a test email from MyRock Mail Engine.', 'myrock-mail-engine' ),
            ],
        ] );

        if ( is_wp_error( $response ) ) {
            wp_send_json_error( [ 'message' => $response->get_error_message() ] );
        }

        $code = wp_remote_retrieve_response_code( $response );
        if ( $code >= 200 && $code < 300 ) {
            wp_send_json_success( [ 'message' => __( 'Test email sent via Mailgun successfully!', 'myrock-mail-engine' ) ] );
        }

        $body    = wp_remote_retrieve_body( $response );
        $decoded = json_decode( $body, true );
        $msg     = $decoded['message'] ?? sprintf( 'Mailgun HTTP %d', $code );
        wp_send_json_error( [ 'message' => $msg ] );
    }
}
