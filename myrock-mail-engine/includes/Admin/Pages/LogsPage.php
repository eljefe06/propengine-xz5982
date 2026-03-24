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
        $current_page = isset( $_GET['paged'] ) ? max( 1, (int) $_GET['paged'] ) : 1;
        $offset       = ( $current_page - 1 ) * self::PER_PAGE;

        // Validate status against a fixed whitelist so it never carries user-controlled data.
        $allowed_statuses = [ 'pending', 'sent', 'failed', 'bounced' ];
        $raw_status       = isset( $_GET['status'] ) ? sanitize_key( $_GET['status'] ) : '';
        $status           = in_array( $raw_status, $allowed_statuses, true ) ? $raw_status : '';

        /*
         * Use fully static SQL with conditional MySQL expressions instead of concatenating
         * a dynamic WHERE clause. This avoids string interpolation of any user-derived variable
         * and satisfies Plugin Check's static-analysis rules.
         *
         * Logic:
         *   (0 = %d OR sl.campaign_id = %d)  — passes all rows when $campaign_id is 0
         *   ('' = %s OR sl.status    = %s)   — passes all rows when $status is ''
         */
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $total_items = (int) $wpdb->get_var(
            $wpdb->prepare(
                'SELECT COUNT(*) FROM %i sl
                 WHERE (0 = %d OR sl.campaign_id = %d)
                   AND (\'\'   = %s OR sl.status    = %s)',
                $logs_table,
                $campaign_id, $campaign_id,
                $status, $status
            )
        );

        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $logs = $wpdb->get_results(
            $wpdb->prepare(
                'SELECT sl.*, c.email AS contact_email, cam.title AS campaign_title
                 FROM %i sl
                 LEFT JOIN %i c   ON c.id   = sl.contact_id
                 LEFT JOIN %i cam ON cam.id = sl.campaign_id
                 WHERE (0 = %d OR sl.campaign_id = %d)
                   AND (\'\'   = %s OR sl.status    = %s)
                 ORDER BY sl.id DESC
                 LIMIT %d OFFSET %d',
                $logs_table, $contacts_table, $campaigns_table,
                $campaign_id, $campaign_id,
                $status, $status,
                self::PER_PAGE, $offset
            ),
            ARRAY_A
        );

        if ( null === $logs ) {
            $logs = [];
        }

        $total_pages = (int) ceil( $total_items / self::PER_PAGE );

        // Fetch all campaigns for the filter dropdown.
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $all_campaigns = $wpdb->get_results(
            "SELECT id, title FROM {$campaigns_table} ORDER BY title ASC",
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
                    esc_html( $row['campaign_title'] ?? '' )
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
