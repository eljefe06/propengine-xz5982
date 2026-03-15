<?php
namespace MyRock\MailEngine\Admin\Pages;

defined('ABSPATH') || exit;


/**
 * Admin Lists page for MyRock Mail Engine.
 *
 * Handles CRUD operations for mailing lists using $wpdb directly.
 */
class ListsPage {

    /**
     * Render the lists page.
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
            global $wpdb;
            $table = $wpdb->prefix . 'mrme_lists';

            // phpcs:ignore WordPress.DB.DirectDatabaseQuery
            $list = $wpdb->get_row(
                $wpdb->prepare( "SELECT * FROM {$table} WHERE id = %d", $id ),
                ARRAY_A
            );

            if ( ! $list ) {
                wp_die( esc_html__( 'List not found.', 'myrock-mail-engine' ) );
            }

            $template = MRME_DIR . 'templates/admin/list-edit.php';
            if ( file_exists( $template ) ) {
                include $template;
            } else {
                $this->render_edit_form( $list, $notice );
            }
            return;
        }

        if ( 'new' === $action ) {
            $list = null;

            $template = MRME_DIR . 'templates/admin/list-edit.php';
            if ( file_exists( $template ) ) {
                include $template;
            } else {
                $this->render_edit_form( null, $notice );
            }
            return;
        }

        // Default: list view.
        $this->render_list( $notice );
    }

    /**
     * Render the mailing lists overview table.
     *
     * @param array|null $notice Optional notice to display.
     * @return void
     */
    private function render_list( ?array $notice ): void {
        global $wpdb;

        $table        = $wpdb->prefix . 'mrme_lists';
        $contacts_tbl = $wpdb->prefix . 'mrme_contact_lists';

        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $lists = $wpdb->get_results(
            "SELECT l.*, COUNT(cl.contact_id) AS subscriber_count
             FROM {$table} l
             LEFT JOIN {$contacts_tbl} cl ON cl.list_id = l.id
             GROUP BY l.id
             ORDER BY l.name ASC",
            ARRAY_A
        );

        if ( null === $lists ) {
            $lists = [];
        }

        $template = MRME_DIR . 'templates/admin/lists-list.php';
        if ( file_exists( $template ) ) {
            include $template;
            return;
        }

        // Fallback inline output.
        echo '<div class="wrap">';
        echo '<h1 class="wp-heading-inline">' . esc_html__( 'Mailing Lists', 'myrock-mail-engine' ) . '</h1>';
        echo '<a href="' . esc_url( admin_url( 'admin.php?page=mrme-lists&action=new' ) ) . '" class="page-title-action">' . esc_html__( 'Add New', 'myrock-mail-engine' ) . '</a>';

        if ( $notice ) {
            printf( '<div class="notice notice-%s is-dismissible"><p>%s</p></div>', esc_attr( $notice['type'] ), esc_html( $notice['message'] ) );
        }

        echo '<table class="wp-list-table widefat fixed striped"><thead><tr>';
        echo '<th>' . esc_html__( 'Name', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Slug', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Subscribers', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Public', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Actions', 'myrock-mail-engine' ) . '</th>';
        echo '</tr></thead><tbody>';

        foreach ( $lists as $row ) {
            $edit_url   = admin_url( 'admin.php?page=mrme-lists&action=edit&id=' . (int) $row['id'] );
            $delete_url = wp_nonce_url(
                admin_url( 'admin-post.php?action=mrme_delete_list&id=' . (int) $row['id'] ),
                'mrme_delete_list_' . (int) $row['id']
            );

            echo '<tr>';
            printf( '<td><a href="%s">%s</a></td>', esc_url( $edit_url ), esc_html( $row['name'] ) );
            echo '<td>' . esc_html( $row['slug'] ) . '</td>';
            echo '<td>' . esc_html( $row['subscriber_count'] ) . '</td>';
            echo '<td>' . ( $row['is_public'] ? esc_html__( 'Yes', 'myrock-mail-engine' ) : esc_html__( 'No', 'myrock-mail-engine' ) ) . '</td>';
            printf(
                '<td><a href="%s">%s</a> | <a href="%s" onclick="return confirm(\'%s\')">%s</a></td>',
                esc_url( $edit_url ),
                esc_html__( 'Edit', 'myrock-mail-engine' ),
                esc_url( $delete_url ),
                esc_js( __( 'Are you sure you want to delete this list?', 'myrock-mail-engine' ) ),
                esc_html__( 'Delete', 'myrock-mail-engine' )
            );
            echo '</tr>';
        }

        echo '</tbody></table>';
        echo '</div>';
    }

    /**
     * Render the inline edit/create form (fallback when template is missing).
     *
     * @param array|null $list   Existing list row, or null for a new list.
     * @param array|null $notice Optional notice to display.
     * @return void
     */
    private function render_edit_form( ?array $list, ?array $notice ): void {
        $is_new     = null === $list;
        $page_title = $is_new ? __( 'Add New List', 'myrock-mail-engine' ) : __( 'Edit List', 'myrock-mail-engine' );
        $list_id    = $is_new ? 0 : (int) $list['id'];

        echo '<div class="wrap">';
        echo '<h1>' . esc_html( $page_title ) . '</h1>';

        if ( $notice ) {
            printf( '<div class="notice notice-%s is-dismissible"><p>%s</p></div>', esc_attr( $notice['type'] ), esc_html( $notice['message'] ) );
        }

        echo '<form method="post" action="' . esc_url( admin_url( 'admin-post.php' ) ) . '">';
        echo '<input type="hidden" name="action" value="mrme_save_list">';
        printf( '<input type="hidden" name="list_id" value="%d">', $list_id );
        wp_nonce_field( 'mrme_save_list' );

        echo '<table class="form-table"><tbody>';

        // Name.
        printf(
            '<tr><th><label for="list_name">%s</label></th><td><input type="text" id="list_name" name="name" value="%s" class="regular-text" required></td></tr>',
            esc_html__( 'Name', 'myrock-mail-engine' ),
            esc_attr( $list['name'] ?? '' )
        );

        // Slug.
        printf(
            '<tr><th><label for="list_slug">%s</label></th><td><input type="text" id="list_slug" name="slug" value="%s" class="regular-text"></td></tr>',
            esc_html__( 'Slug', 'myrock-mail-engine' ),
            esc_attr( $list['slug'] ?? '' )
        );

        // Description.
        printf(
            '<tr><th><label for="list_description">%s</label></th><td><textarea id="list_description" name="description" rows="4" class="large-text">%s</textarea></td></tr>',
            esc_html__( 'Description', 'myrock-mail-engine' ),
            esc_textarea( $list['description'] ?? '' )
        );

        // Is Public.
        $is_public_checked = ! empty( $list['is_public'] ) ? 'checked' : '';
        printf(
            '<tr><th>%s</th><td><label><input type="checkbox" name="is_public" value="1" %s> %s</label></td></tr>',
            esc_html__( 'Public', 'myrock-mail-engine' ),
            esc_attr( $is_public_checked ),
            esc_html__( 'Allow visitors to subscribe via signup forms', 'myrock-mail-engine' )
        );

        echo '</tbody></table>';

        submit_button( $is_new ? __( 'Create List', 'myrock-mail-engine' ) : __( 'Update List', 'myrock-mail-engine' ) );

        echo '</form>';
        echo '</div>';
    }

    /**
     * Handle saving a list (create or update).
     *
     * @return void
     */
    public function handle_save(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to perform this action.', 'myrock-mail-engine' ) );
        }

        check_admin_referer( 'mrme_save_list' );

        global $wpdb;
        $table = $wpdb->prefix . 'mrme_lists';

        $list_id     = isset( $_POST['list_id'] ) ? (int) $_POST['list_id'] : 0;
        $name        = isset( $_POST['name'] ) ? sanitize_text_field( wp_unslash( $_POST['name'] ) ) : '';
        $slug        = isset( $_POST['slug'] ) ? sanitize_title( wp_unslash( $_POST['slug'] ) ) : '';
        $description = isset( $_POST['description'] ) ? sanitize_textarea_field( wp_unslash( $_POST['description'] ) ) : '';
        $is_public   = isset( $_POST['is_public'] ) ? 1 : 0;

        $redirect = admin_url( 'admin.php?page=mrme-lists' );

        if ( ! $name ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'name_required', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        // Generate slug from name if not provided.
        if ( ! $slug ) {
            $slug = sanitize_title( $name );
        }

        // Ensure slug uniqueness.
        $slug = $this->unique_slug( $slug, $list_id );

        $data = [
            'name'        => $name,
            'slug'        => $slug,
            'description' => $description,
            'is_public'   => $is_public,
        ];

        $format = [ '%s', '%s', '%s', '%d' ];

        if ( $list_id > 0 ) {
            // phpcs:ignore WordPress.DB.DirectDatabaseQuery
            $result = $wpdb->update( $table, $data, [ 'id' => $list_id ], $format, [ '%d' ] );
            $action = 'updated';
        } else {
            $data['created_at'] = current_time( 'mysql' );
            $format[]           = '%s';

            // phpcs:ignore WordPress.DB.DirectDatabaseQuery
            $result  = $wpdb->insert( $table, $data, $format );
            $list_id = (int) $wpdb->insert_id;
            $action  = 'created';
        }

        if ( false === $result ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'save_failed', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        $redirect = admin_url( 'admin.php?page=mrme-lists&action=edit&id=' . $list_id );
        wp_safe_redirect( add_query_arg( [ 'notice' => $action, 'notice_type' => 'success' ], $redirect ) );
        die();
    }

    /**
     * Generate a unique slug for a list.
     *
     * @param string $slug    Proposed slug.
     * @param int    $list_id List ID to exclude from uniqueness check (for updates).
     * @return string Unique slug.
     */
    private function unique_slug( string $slug, int $list_id = 0 ): string {
        global $wpdb;
        $table     = $wpdb->prefix . 'mrme_lists';
        $original  = $slug;
        $counter   = 1;

        while ( true ) {
            if ( $list_id > 0 ) {
                // phpcs:ignore WordPress.DB.DirectDatabaseQuery
                $existing = $wpdb->get_var(
                    $wpdb->prepare(
                        "SELECT id FROM {$table} WHERE slug = %s AND id != %d LIMIT 1",
                        $slug,
                        $list_id
                    )
                );
            } else {
                // phpcs:ignore WordPress.DB.DirectDatabaseQuery
                $existing = $wpdb->get_var(
                    $wpdb->prepare(
                        "SELECT id FROM {$table} WHERE slug = %s LIMIT 1",
                        $slug
                    )
                );
            }

            if ( ! $existing ) {
                break;
            }

            $slug = $original . '-' . $counter;
            $counter++;
        }

        return $slug;
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
            'created'       => __( 'List created successfully.', 'myrock-mail-engine' ),
            'updated'       => __( 'List updated successfully.', 'myrock-mail-engine' ),
            'deleted'       => __( 'List deleted successfully.', 'myrock-mail-engine' ),
            'save_failed'   => __( 'Failed to save list. Please try again.', 'myrock-mail-engine' ),
            'delete_failed' => __( 'Failed to delete list. Please try again.', 'myrock-mail-engine' ),
            'name_required' => __( 'List name is required.', 'myrock-mail-engine' ),
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
