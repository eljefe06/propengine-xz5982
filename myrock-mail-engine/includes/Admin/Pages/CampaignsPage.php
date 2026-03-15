<?php
namespace MyRock\MailEngine\Admin\Pages;

defined('ABSPATH') || exit;


use MyRock\MailEngine\Models\Campaign;
use MyRock\MailEngine\Services\CampaignService;

/**
 * Admin Campaigns page for MyRock Mail Engine.
 */
class CampaignsPage {

    /**
     * Render the campaigns page.
     *
     * Dispatches to the edit form or list view based on query parameters.
     *
     * @return void
     */
    public function render(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to view this page.', 'myrock-mail-engine' ) );
        }

        $action = isset( $_GET['action'] ) ? sanitize_key( $_GET['action'] ) : 'list';
        $id     = isset( $_GET['id'] ) ? (int) $_GET['id'] : 0;
        $notice = $this->get_notice();

        if ( 'edit' === $action && $id > 0 ) {
            $campaign = Campaign::find( $id );

            if ( ! $campaign ) {
                wp_die( esc_html__( 'Campaign not found.', 'myrock-mail-engine' ) );
            }

            $lists = $this->get_all_lists();
            $tags  = $this->get_all_tags();

            $template = MRME_DIR . 'templates/admin/campaign-edit.php';
            if ( file_exists( $template ) ) {
                include $template;
            } else {
                $this->render_edit_form( $campaign, $lists, $tags, $notice );
            }
            return;
        }

        if ( 'new' === $action ) {
            $campaign = null;
            $lists    = $this->get_all_lists();
            $tags     = $this->get_all_tags();

            $template = MRME_DIR . 'templates/admin/campaign-edit.php';
            if ( file_exists( $template ) ) {
                include $template;
            } else {
                $this->render_edit_form( null, $lists, $tags, $notice );
            }
            return;
        }

        // Default: list view.
        $this->render_list( $notice );
    }

    /**
     * Render the campaigns list view.
     *
     * @param array|null $notice Optional notice to display.
     * @return void
     */
    private function render_list( ?array $notice ): void {
        global $wpdb;

        $table = $wpdb->prefix . 'mrme_campaigns';

        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $campaigns = $wpdb->get_results(
            "SELECT * FROM {$table} ORDER BY created_at DESC",
            ARRAY_A
        );

        if ( null === $campaigns ) {
            $campaigns = [];
        }

        $template = MRME_DIR . 'templates/admin/campaigns-list.php';
        if ( file_exists( $template ) ) {
            include $template;
            return;
        }

        // Fallback inline output.
        echo '<div class="wrap">';
        echo '<h1 class="wp-heading-inline">' . esc_html__( 'Campaigns', 'myrock-mail-engine' ) . '</h1>';
        echo '<a href="' . esc_url( admin_url( 'admin.php?page=mrme-campaigns&action=new' ) ) . '" class="page-title-action">' . esc_html__( 'Add New', 'myrock-mail-engine' ) . '</a>';

        if ( $notice ) {
            printf( '<div class="notice notice-%s is-dismissible"><p>%s</p></div>', esc_attr( $notice['type'] ), esc_html( $notice['message'] ) );
        }

        echo '<table class="wp-list-table widefat fixed striped"><thead><tr>';
        echo '<th>' . esc_html__( 'Name', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Subject', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Status', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Scheduled At', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Actions', 'myrock-mail-engine' ) . '</th>';
        echo '</tr></thead><tbody>';

        foreach ( $campaigns as $row ) {
            $edit_url   = admin_url( 'admin.php?page=mrme-campaigns&action=edit&id=' . (int) $row['id'] );
            $delete_url = wp_nonce_url(
                admin_url( 'admin-post.php?action=mrme_delete_campaign&id=' . (int) $row['id'] ),
                'mrme_delete_campaign_' . (int) $row['id']
            );
            $send_url = wp_nonce_url(
                admin_url( 'admin-post.php?action=mrme_send_campaign&id=' . (int) $row['id'] ),
                'mrme_send_campaign_' . (int) $row['id']
            );

            echo '<tr>';
            printf( '<td><a href="%s">%s</a></td>', esc_url( $edit_url ), esc_html( $row['name'] ) );
            echo '<td>' . esc_html( $row['subject'] ) . '</td>';
            echo '<td>' . esc_html( $row['status'] ) . '</td>';
            echo '<td>' . esc_html( $row['scheduled_at'] ?? '—' ) . '</td>';

            echo '<td>';
            printf( '<a href="%s">%s</a>', esc_url( $edit_url ), esc_html__( 'Edit', 'myrock-mail-engine' ) );

            if ( in_array( $row['status'], [ 'draft', 'scheduled' ], true ) ) {
                printf(
                    ' | <a href="%s" onclick="return confirm(\'%s\')">%s</a>',
                    esc_url( $send_url ),
                    esc_js( __( 'Send this campaign now?', 'myrock-mail-engine' ) ),
                    esc_html__( 'Send Now', 'myrock-mail-engine' )
                );
            }

            printf(
                ' | <a href="%s" onclick="return confirm(\'%s\')">%s</a>',
                esc_url( $delete_url ),
                esc_js( __( 'Are you sure you want to delete this campaign?', 'myrock-mail-engine' ) ),
                esc_html__( 'Delete', 'myrock-mail-engine' )
            );
            echo '</td>';

            echo '</tr>';
        }

        echo '</tbody></table>';
        echo '</div>';
    }

    /**
     * Render the inline edit/create form (fallback when template is missing).
     *
     * @param Campaign|object|array|null $campaign Existing campaign, or null.
     * @param array                      $lists    All available lists.
     * @param array                      $tags     All available tags.
     * @param array|null                 $notice   Optional notice to display.
     * @return void
     */
    private function render_edit_form( $campaign, array $lists, array $tags, ?array $notice ): void {
        $is_new    = null === $campaign;
        $page_title = $is_new ? __( 'Create Campaign', 'myrock-mail-engine' ) : __( 'Edit Campaign', 'myrock-mail-engine' );

        $campaign_id   = $is_new ? 0 : ( is_array( $campaign ) ? (int) $campaign['id'] : (int) $campaign->id );
        $name          = $is_new ? '' : ( is_array( $campaign ) ? $campaign['name'] : $campaign->name );
        $subject       = $is_new ? '' : ( is_array( $campaign ) ? $campaign['subject'] : $campaign->subject );
        $preview_text  = $is_new ? '' : ( is_array( $campaign ) ? $campaign['preview_text'] : $campaign->preview_text );
        $body_html     = $is_new ? '' : ( is_array( $campaign ) ? $campaign['body_html'] : $campaign->body_html );
        $body_text     = $is_new ? '' : ( is_array( $campaign ) ? $campaign['body_text'] : $campaign->body_text );
        $from_name     = $is_new ? '' : ( is_array( $campaign ) ? $campaign['from_name'] : $campaign->from_name );
        $from_email    = $is_new ? '' : ( is_array( $campaign ) ? $campaign['from_email'] : $campaign->from_email );
        $reply_to      = $is_new ? '' : ( is_array( $campaign ) ? $campaign['reply_to'] : $campaign->reply_to );
        $list_ids      = $is_new ? [] : explode( ',', ( is_array( $campaign ) ? $campaign['list_ids'] : $campaign->list_ids ) );
        $tag_ids       = $is_new ? [] : explode( ',', ( is_array( $campaign ) ? $campaign['tag_ids'] : $campaign->tag_ids ) );
        $scheduled_at  = $is_new ? '' : ( is_array( $campaign ) ? $campaign['scheduled_at'] : $campaign->scheduled_at );
        $status        = $is_new ? 'draft' : ( is_array( $campaign ) ? $campaign['status'] : $campaign->status );

        echo '<div class="wrap">';
        echo '<h1>' . esc_html( $page_title ) . '</h1>';

        if ( $notice ) {
            printf( '<div class="notice notice-%s is-dismissible"><p>%s</p></div>', esc_attr( $notice['type'] ), esc_html( $notice['message'] ) );
        }

        echo '<form method="post" action="' . esc_url( admin_url( 'admin-post.php' ) ) . '">';
        echo '<input type="hidden" name="action" value="mrme_save_campaign">';
        printf( '<input type="hidden" name="campaign_id" value="%d">', $campaign_id );
        wp_nonce_field( 'mrme_save_campaign' );

        echo '<table class="form-table"><tbody>';

        printf(
            '<tr><th><label for="campaign_name">%s</label></th><td><input type="text" id="campaign_name" name="name" value="%s" class="regular-text" required></td></tr>',
            esc_html__( 'Campaign Name', 'myrock-mail-engine' ),
            esc_attr( $name )
        );

        printf(
            '<tr><th><label for="campaign_subject">%s</label></th><td><input type="text" id="campaign_subject" name="subject" value="%s" class="regular-text" required></td></tr>',
            esc_html__( 'Subject Line', 'myrock-mail-engine' ),
            esc_attr( $subject )
        );

        printf(
            '<tr><th><label for="preview_text">%s</label></th><td><input type="text" id="preview_text" name="preview_text" value="%s" class="regular-text"></td></tr>',
            esc_html__( 'Preview Text', 'myrock-mail-engine' ),
            esc_attr( $preview_text )
        );

        printf(
            '<tr><th><label for="from_name">%s</label></th><td><input type="text" id="from_name" name="from_name" value="%s" class="regular-text"></td></tr>',
            esc_html__( 'From Name', 'myrock-mail-engine' ),
            esc_attr( $from_name )
        );

        printf(
            '<tr><th><label for="from_email">%s</label></th><td><input type="email" id="from_email" name="from_email" value="%s" class="regular-text"></td></tr>',
            esc_html__( 'From Email', 'myrock-mail-engine' ),
            esc_attr( $from_email )
        );

        printf(
            '<tr><th><label for="reply_to">%s</label></th><td><input type="email" id="reply_to" name="reply_to" value="%s" class="regular-text"></td></tr>',
            esc_html__( 'Reply-To', 'myrock-mail-engine' ),
            esc_attr( $reply_to )
        );

        // HTML Body.
        echo '<tr><th><label for="body_html">' . esc_html__( 'HTML Body', 'myrock-mail-engine' ) . '</label></th><td>';
        wp_editor(
            $body_html,
            'body_html',
            [
                'textarea_name' => 'body_html',
                'media_buttons' => true,
                'teeny'         => false,
                'tinymce'       => true,
                'quicktags'     => true,
            ]
        );
        echo '</td></tr>';

        // Plain Text Body.
        printf(
            '<tr><th><label for="body_text">%s</label></th><td><textarea id="body_text" name="body_text" rows="8" class="large-text">%s</textarea></td></tr>',
            esc_html__( 'Plain Text Body', 'myrock-mail-engine' ),
            esc_textarea( $body_text )
        );

        // Lists.
        echo '<tr><th>' . esc_html__( 'Send To Lists', 'myrock-mail-engine' ) . '</th><td>';
        foreach ( $lists as $lst ) {
            $lst_id      = is_array( $lst ) ? (int) $lst['id'] : (int) $lst->id;
            $lst_name    = is_array( $lst ) ? $lst['name'] : $lst->name;
            $is_selected = in_array( (string) $lst_id, array_map( 'strval', $list_ids ), true );
            printf(
                '<label><input type="checkbox" name="list_ids[]" value="%d" %s> %s</label><br>',
                $lst_id,
                checked( $is_selected, true, false ),
                esc_html( $lst_name )
            );
        }
        echo '</td></tr>';

        // Tags.
        echo '<tr><th>' . esc_html__( 'Filter by Tags', 'myrock-mail-engine' ) . '</th><td>';
        foreach ( $tags as $tag ) {
            $tag_id      = is_array( $tag ) ? (int) $tag['id'] : (int) $tag->id;
            $tag_name    = is_array( $tag ) ? $tag['name'] : $tag->name;
            $is_selected = in_array( (string) $tag_id, array_map( 'strval', $tag_ids ), true );
            printf(
                '<label><input type="checkbox" name="tag_ids[]" value="%d" %s> %s</label><br>',
                $tag_id,
                checked( $is_selected, true, false ),
                esc_html( $tag_name )
            );
        }
        echo '</td></tr>';

        // Scheduled At.
        printf(
            '<tr><th><label for="scheduled_at">%s</label></th><td><input type="datetime-local" id="scheduled_at" name="scheduled_at" value="%s"><p class="description">%s</p></td></tr>',
            esc_html__( 'Schedule Send', 'myrock-mail-engine' ),
            esc_attr( $scheduled_at ? date( 'Y-m-d\TH:i', strtotime( $scheduled_at ) ) : '' ),
            esc_html__( 'Leave blank to save as draft. Set a date/time to schedule.', 'myrock-mail-engine' )
        );

        echo '</tbody></table>';

        submit_button( $is_new ? __( 'Save Campaign', 'myrock-mail-engine' ) : __( 'Update Campaign', 'myrock-mail-engine' ) );

        // Test send section (only for existing campaigns).
        if ( ! $is_new ) {
            echo '<hr>';
            echo '<h2>' . esc_html__( 'Send Test Email', 'myrock-mail-engine' ) . '</h2>';
            echo '<p>';
            printf(
                '<input type="email" id="mrme_test_email" placeholder="%s" class="regular-text">',
                esc_attr__( 'test@example.com', 'myrock-mail-engine' )
            );
            printf(
                ' <button type="button" class="button button-secondary" id="mrme_send_test" data-campaign="%d">%s</button>',
                $campaign_id,
                esc_html__( 'Send Test', 'myrock-mail-engine' )
            );
            echo '</p>';
            echo '<p id="mrme_test_result"></p>';
        }

        echo '</form>';
        echo '</div>';
    }

    /**
     * Handle saving a campaign (create or update).
     *
     * Sets status to 'scheduled' when a scheduled_at date is provided and status
     * is not explicitly 'draft'. Otherwise saves as 'draft'.
     *
     * @return void
     */
    public function handle_save(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to perform this action.', 'myrock-mail-engine' ) );
        }

        check_admin_referer( 'mrme_save_campaign' );

        $campaign_id  = isset( $_POST['campaign_id'] ) ? (int) $_POST['campaign_id'] : 0;
        $name         = isset( $_POST['name'] ) ? sanitize_text_field( wp_unslash( $_POST['name'] ) ) : '';
        $subject      = isset( $_POST['subject'] ) ? sanitize_text_field( wp_unslash( $_POST['subject'] ) ) : '';
        $preview_text = isset( $_POST['preview_text'] ) ? sanitize_text_field( wp_unslash( $_POST['preview_text'] ) ) : '';
        $from_name    = isset( $_POST['from_name'] ) ? sanitize_text_field( wp_unslash( $_POST['from_name'] ) ) : '';
        $from_email   = isset( $_POST['from_email'] ) ? sanitize_email( wp_unslash( $_POST['from_email'] ) ) : '';
        $reply_to     = isset( $_POST['reply_to'] ) ? sanitize_email( wp_unslash( $_POST['reply_to'] ) ) : '';
        $body_html    = isset( $_POST['body_html'] ) ? wp_kses_post( wp_unslash( $_POST['body_html'] ) ) : '';
        $body_text    = isset( $_POST['body_text'] ) ? sanitize_textarea_field( wp_unslash( $_POST['body_text'] ) ) : '';
        $scheduled_at = isset( $_POST['scheduled_at'] ) ? sanitize_text_field( wp_unslash( $_POST['scheduled_at'] ) ) : '';

        // List IDs as comma-separated string.
        $list_ids_raw = isset( $_POST['list_ids'] ) && is_array( $_POST['list_ids'] )
            ? array_map( 'intval', $_POST['list_ids'] )
            : [];
        $list_ids = implode( ',', array_filter( $list_ids_raw ) );

        // Tag IDs as comma-separated string.
        $tag_ids_raw = isset( $_POST['tag_ids'] ) && is_array( $_POST['tag_ids'] )
            ? array_map( 'intval', $_POST['tag_ids'] )
            : [];
        $tag_ids = implode( ',', array_filter( $tag_ids_raw ) );

        $redirect = admin_url( 'admin.php?page=mrme-campaigns' );

        if ( ! $name || ! $subject ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'required_fields', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        // Normalize scheduled_at to MySQL datetime.
        $scheduled_at_mysql = '';
        if ( $scheduled_at ) {
            $ts = strtotime( $scheduled_at );
            if ( $ts ) {
                $scheduled_at_mysql = date( 'Y-m-d H:i:s', $ts );
            }
        }

        // Determine status.
        if ( $scheduled_at_mysql ) {
            $status = 'scheduled';
        } else {
            $status = 'draft';
        }

        $data = [
            'name'         => $name,
            'subject'      => $subject,
            'preview_text' => $preview_text,
            'from_name'    => $from_name,
            'from_email'   => $from_email,
            'reply_to'     => $reply_to,
            'body_html'    => $body_html,
            'body_text'    => $body_text,
            'list_ids'     => $list_ids,
            'tag_ids'      => $tag_ids,
            'scheduled_at' => $scheduled_at_mysql ?: null,
            'status'       => $status,
        ];

        if ( $campaign_id > 0 ) {
            $result = Campaign::update( $campaign_id, $data );
            $action = 'updated';
        } else {
            $result    = Campaign::create( $data );
            $action    = 'created';

            if ( $result ) {
                $campaign_id = (int) $result;
            }
        }

        if ( false === $result || is_wp_error( $result ) ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'save_failed', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        $redirect = admin_url( 'admin.php?page=mrme-campaigns&action=edit&id=' . $campaign_id );
        wp_safe_redirect( add_query_arg( [ 'notice' => $action, 'notice_type' => 'success' ], $redirect ) );
        die();
    }

    /**
     * Handle sending a campaign immediately.
     *
     * @return void
     */
    public function handle_send(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to perform this action.', 'myrock-mail-engine' ) );
        }

        $id = isset( $_GET['id'] ) ? (int) $_GET['id'] : 0;

        if ( ! $id ) {
            wp_die( esc_html__( 'Invalid campaign ID.', 'myrock-mail-engine' ) );
        }

        check_admin_referer( 'mrme_send_campaign_' . $id );

        $redirect = admin_url( 'admin.php?page=mrme-campaigns' );

        $service = new CampaignService();
        $result  = $service->send_now( $id );

        if ( is_wp_error( $result ) ) {
            wp_safe_redirect(
                add_query_arg(
                    [ 'notice' => 'send_failed', 'notice_type' => 'error' ],
                    admin_url( 'admin.php?page=mrme-campaigns&action=edit&id=' . $id )
                )
            );
            die();
        }

        wp_safe_redirect( add_query_arg( [ 'notice' => 'sent', 'notice_type' => 'success' ], $redirect ) );
        die();
    }

    /**
     * Fetch all mailing lists.
     *
     * @return array
     */
    private function get_all_lists(): array {
        global $wpdb;
        $table = $wpdb->prefix . 'mrme_lists';
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $results = $wpdb->get_results( "SELECT id, name FROM {$table} ORDER BY name ASC", ARRAY_A );
        return $results ?: [];
    }

    /**
     * Fetch all tags.
     *
     * @return array
     */
    private function get_all_tags(): array {
        global $wpdb;
        $table = $wpdb->prefix . 'mrme_tags';
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $results = $wpdb->get_results( "SELECT id, name FROM {$table} ORDER BY name ASC", ARRAY_A );
        return $results ?: [];
    }

    /**
     * Retrieve and format a notice from query params.
     *
     * @return array|null Array with 'type' and 'message' keys, or null if none.
     */
    private function get_notice(): ?array {
        $notice_key  = isset( $_GET['notice'] ) ? sanitize_key( $_GET['notice'] ) : '';
        $notice_type = isset( $_GET['notice_type'] ) ? sanitize_key( $_GET['notice_type'] ) : 'success';

        if ( ! $notice_key ) {
            return null;
        }

        $messages = [
            'created'        => __( 'Campaign created successfully.', 'myrock-mail-engine' ),
            'updated'        => __( 'Campaign updated successfully.', 'myrock-mail-engine' ),
            'deleted'        => __( 'Campaign deleted successfully.', 'myrock-mail-engine' ),
            'sent'           => __( 'Campaign is being sent.', 'myrock-mail-engine' ),
            'save_failed'    => __( 'Failed to save campaign. Please try again.', 'myrock-mail-engine' ),
            'send_failed'    => __( 'Failed to send campaign. Please try again.', 'myrock-mail-engine' ),
            'delete_failed'  => __( 'Failed to delete campaign.', 'myrock-mail-engine' ),
            'required_fields' => __( 'Campaign name and subject are required.', 'myrock-mail-engine' ),
        ];

        $message = isset( $messages[ $notice_key ] ) ? $messages[ $notice_key ] : '';

        if ( ! $message ) {
            return null;
        }

        return [
            'type'    => $notice_type,
            'message' => $message,
        ];
    }
}
