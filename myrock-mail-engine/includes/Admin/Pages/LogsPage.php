<?php
namespace MyRock\MailEngine\Admin\Pages;

defined('ABSPATH') || exit;


use MyRock\MailEngine\Models\SendLog;

/**
 * Admin Logs page for MyRock Mail Engine.
 *
 * Displays send log entries with filtering by campaign and status.
 */
class LogsPage {

    /**
     * Number of log rows to display per page.
     *
     * @var int
     */
    private const PER_PAGE = 50;

    /**
     * Render the logs page.
     *
     * @return void
     */
    public function render(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to view this page.', 'myrock-mail-engine' ) );
        }

        global $wpdb;

        $logs_table      = $wpdb->prefix . 'mrme_send_logs';
        $contacts_table  = $wpdb->prefix . 'mrme_contacts';
        $campaigns_table = $wpdb->prefix . 'mrme_campaigns';

        // Filters.
        $campaign_id  = isset( $_GET['campaign_id'] ) ? (int) $_GET['campaign_id'] : 0;
        $status       = isset( $_GET['status'] ) ? sanitize_key( $_GET['status'] ) : '';
        $current_page = isset( $_GET['paged'] ) ? max( 1, (int) $_GET['paged'] ) : 1;
        $offset       = ( $current_page - 1 ) * self::PER_PAGE;

        // Build WHERE clauses.
        $where_clauses = [];
        $placeholders  = [];

        if ( $campaign_id > 0 ) {
            $where_clauses[] = 'sl.campaign_id = %d';
            $placeholders[]  = $campaign_id;
        }

        if ( $status ) {
            $where_clauses[] = 'sl.status = %s';
            $placeholders[]  = $status;
        }

        $where_sql = $where_clauses ? 'WHERE ' . implode( ' AND ', $where_clauses ) : '';

        $count_sql = "SELECT COUNT(*) FROM {$logs_table} sl {$where_sql}";
        $list_sql  = "SELECT sl.*, c.email AS contact_email, cam.name AS campaign_name
                      FROM {$logs_table} sl
                      LEFT JOIN {$contacts_table} c ON c.id = sl.contact_id
                      LEFT JOIN {$campaigns_table} cam ON cam.id = sl.campaign_id
                      {$where_sql}
                      ORDER BY sl.sent_at DESC
                      LIMIT %d OFFSET %d";

        if ( $placeholders ) {
            // phpcs:ignore WordPress.DB.DirectDatabaseQuery, WordPress.DB.PreparedSQL.NotPrepared
            $total_items = (int) $wpdb->get_var( $wpdb->prepare( $count_sql, $placeholders ) );

            $list_placeholders = array_merge( $placeholders, [ self::PER_PAGE, $offset ] );
            // phpcs:ignore WordPress.DB.DirectDatabaseQuery, WordPress.DB.PreparedSQL.NotPrepared
            $logs = $wpdb->get_results( $wpdb->prepare( $list_sql, $list_placeholders ), ARRAY_A );
        } else {
            // phpcs:ignore WordPress.DB.DirectDatabaseQuery, WordPress.DB.PreparedSQL.InterpolatedNotPrepared
            $total_items = (int) $wpdb->get_var( $count_sql );
            // phpcs:ignore WordPress.DB.DirectDatabaseQuery, WordPress.DB.PreparedSQL.InterpolatedNotPrepared
            $logs = $wpdb->get_results( $wpdb->prepare( $list_sql, self::PER_PAGE, $offset ), ARRAY_A );
        }

        if ( null === $logs ) {
            $logs = [];
        }

        $total_pages = (int) ceil( $total_items / self::PER_PAGE );

        // Fetch all campaigns for the filter dropdown.
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $all_campaigns = $wpdb->get_results(
            "SELECT id, name FROM {$campaigns_table} ORDER BY name ASC",
            ARRAY_A
        );

        if ( null === $all_campaigns ) {
            $all_campaigns = [];
        }

        $available_statuses = [
            'pending'  => __( 'Pending', 'myrock-mail-engine' ),
            'sent'     => __( 'Sent', 'myrock-mail-engine' ),
            'failed'   => __( 'Failed', 'myrock-mail-engine' ),
            'bounced'  => __( 'Bounced', 'myrock-mail-engine' ),
        ];

        $template = MRME_DIR . 'templates/admin/logs-list.php';
        if ( file_exists( $template ) ) {
            include $template;
            return;
        }

        // Fallback inline output.
        echo '<div class="wrap">';
        echo '<h1>' . esc_html__( 'Send Logs', 'myrock-mail-engine' ) . '</h1>';

        // Filter form.
        echo '<form method="get" action="' . esc_url( admin_url( 'admin.php' ) ) . '">';
        echo '<input type="hidden" name="page" value="mrme-logs">';

        echo '<select name="campaign_id">';
        echo '<option value="">' . esc_html__( '— All Campaigns —', 'myrock-mail-engine' ) . '</option>';
        foreach ( $all_campaigns as $camp ) {
            printf(
                '<option value="%d" %s>%s</option>',
                (int) $camp['id'],
                selected( $campaign_id, (int) $camp['id'], false ),
                esc_html( $camp['name'] )
            );
        }
        echo '</select>';

        echo '<select name="status">';
        echo '<option value="">' . esc_html__( '— All Statuses —', 'myrock-mail-engine' ) . '</option>';
        foreach ( $available_statuses as $val => $label ) {
            printf( '<option value="%s" %s>%s</option>', esc_attr( $val ), selected( $status, $val, false ), esc_html( $label ) );
        }
        echo '</select>';

        submit_button( __( 'Filter', 'myrock-mail-engine' ), 'secondary', 'filter_submit', false );
        echo '</form>';

        // Stats summary.
        printf(
            '<p>' . esc_html__( 'Showing %1$d–%2$d of %3$d entries.', 'myrock-mail-engine' ) . '</p>',
            min( $offset + 1, $total_items ),
            min( $offset + self::PER_PAGE, $total_items ),
            $total_items
        );

        // Table.
        echo '<table class="wp-list-table widefat fixed striped"><thead><tr>';
        echo '<th>' . esc_html__( 'Contact', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Campaign', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Status', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Sent At', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Opened At', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Clicked At', 'myrock-mail-engine' ) . '</th>';
        echo '</tr></thead><tbody>';

        if ( empty( $logs ) ) {
            echo '<tr><td colspan="6">' . esc_html__( 'No log entries found.', 'myrock-mail-engine' ) . '</td></tr>';
        } else {
            foreach ( $logs as $row ) {
                $contact_url = admin_url( 'admin.php?page=mrme-contacts&action=edit&id=' . (int) $row['contact_id'] );
                $campaign_url = admin_url( 'admin.php?page=mrme-campaigns&action=edit&id=' . (int) $row['campaign_id'] );

                echo '<tr>';
                printf(
                    '<td><a href="%s">%s</a></td>',
                    esc_url( $contact_url ),
                    esc_html( $row['contact_email'] ?? '' )
                );
                printf(
                    '<td><a href="%s">%s</a></td>',
                    esc_url( $campaign_url ),
                    esc_html( $row['campaign_name'] ?? '' )
                );
                echo '<td>' . esc_html( $row['status'] ) . '</td>';
                echo '<td>' . esc_html( $row['sent_at'] ?? '—' ) . '</td>';
                echo '<td>' . esc_html( $row['opened_at'] ?? '—' ) . '</td>';
                echo '<td>' . esc_html( $row['clicked_at'] ?? '—' ) . '</td>';
                echo '</tr>';
            }
        }

        echo '</tbody></table>';

        // Pagination.
        if ( $total_pages > 1 ) {
            echo '<div class="tablenav bottom"><div class="tablenav-pages">';

            for ( $p = 1; $p <= $total_pages; $p++ ) {
                $page_url = add_query_arg(
                    [
                        'page'        => 'mrme-logs',
                        'paged'       => $p,
                        'campaign_id' => $campaign_id,
                        'status'      => $status,
                    ],
                    admin_url( 'admin.php' )
                );

                if ( $p === $current_page ) {
                    printf( '<span class="current">%d</span> ', $p );
                } else {
                    printf( '<a href="%s">%d</a> ', esc_url( $page_url ), $p );
                }
            }

            echo '</div></div>';
        }

        echo '</div>';
    }
}
