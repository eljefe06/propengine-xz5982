<?php
namespace MyRock\MailEngine\Admin\Pages;

defined('ABSPATH') || exit;


/**
 * Admin Dashboard page for MyRock Mail Engine.
 */
class DashboardPage {

    /**
     * Render the dashboard page.
     *
     * Gathers summary statistics and loads the dashboard template.
     *
     * @return void
     */
    public function render(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to view this page.', 'myrock-mail-engine' ) );
        }

        global $wpdb;

        $contacts_table  = $wpdb->prefix . 'mrme_contacts';
        $campaigns_table = $wpdb->prefix . 'mrme_campaigns';
        $logs_table      = $wpdb->prefix . 'mrme_send_logs';

        // Total contacts.
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $total_contacts = (int) $wpdb->get_var( "SELECT COUNT(*) FROM {$contacts_table}" );

        // Total campaigns.
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $total_campaigns = (int) $wpdb->get_var( "SELECT COUNT(*) FROM {$campaigns_table}" );

        // Total emails sent (rows in send_logs).
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $total_sent = (int) $wpdb->get_var( "SELECT COUNT(*) FROM {$logs_table}" );

        // Total opens — sum from campaigns table.
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $total_opens = (int) $wpdb->get_var( "SELECT SUM(total_opens) FROM {$campaigns_table}" );

        // Recent campaigns (last 5).
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $recent_campaigns = $wpdb->get_results(
            "SELECT id, title, status, sent_at, scheduled_at, total_sent, total_opens, total_clicks, created_at FROM {$campaigns_table} ORDER BY created_at DESC LIMIT 5",
            ARRAY_A
        );

        if ( null === $recent_campaigns ) {
            $recent_campaigns = [];
        }

        // Recent send log activity (last 10).
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $recent_logs = $wpdb->get_results(
            "SELECT sl.id, sl.status, sl.sent_at, sl.email
             FROM {$logs_table} sl
             ORDER BY sl.id DESC
             LIMIT 10",
            ARRAY_A
        );

        if ( null === $recent_logs ) {
            $recent_logs = [];
        }

        // Pass variables to template.
        $template_data = [
            'total_contacts'   => $total_contacts,
            'total_campaigns'  => $total_campaigns,
            'total_sent'       => $total_sent,
            'total_opens'      => $total_opens,
            'recent_campaigns' => $recent_campaigns,
            'recent_logs'      => $recent_logs,
        ];

        // Extract variables for use in template.
        extract( $template_data, EXTR_SKIP ); // phpcs:ignore WordPress.PHP.DontExtract

        $template = MRME_DIR . 'templates/admin/dashboard.php';

        if ( file_exists( $template ) ) {
            include $template;
        } else {
            // Fallback inline output when template file is not yet present.
            echo '<div class="wrap">';
            echo '<h1>' . esc_html__( 'MyRock Mail Engine — Dashboard', 'myrock-mail-engine' ) . '</h1>';
            echo '<div class="mrme-dashboard-stats">';
            printf(
                '<div class="mrme-stat"><strong>%s</strong><span>%d</span></div>',
                esc_html__( 'Total Contacts', 'myrock-mail-engine' ),
                (int) $total_contacts
            );
            printf(
                '<div class="mrme-stat"><strong>%s</strong><span>%d</span></div>',
                esc_html__( 'Total Campaigns', 'myrock-mail-engine' ),
                (int) $total_campaigns
            );
            printf(
                '<div class="mrme-stat"><strong>%s</strong><span>%d</span></div>',
                esc_html__( 'Emails Sent', 'myrock-mail-engine' ),
                (int) $total_sent
            );
            printf(
                '<div class="mrme-stat"><strong>%s</strong><span>%d</span></div>',
                esc_html__( 'Total Opens', 'myrock-mail-engine' ),
                (int) $total_opens
            );
            echo '</div>';
            echo '</div>';
        }
    }
}
