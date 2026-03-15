<?php
namespace MyRock\MailEngine\Admin\Pages;

defined('ABSPATH') || exit;


use MyRock\MailEngine\Models\Tag;

/**
 * Admin Tags page for MyRock Mail Engine.
 */
class TagsPage {

    /**
     * Render the tags page.
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
            $tag = Tag::find( $id );

            if ( ! $tag ) {
                wp_die( esc_html__( 'Tag not found.', 'myrock-mail-engine' ) );
            }

            $template = MRME_DIR . 'templates/admin/tag-edit.php';
            if ( file_exists( $template ) ) {
                include $template;
            } else {
                $this->render_edit_form( $tag, $notice );
            }
            return;
        }

        if ( 'new' === $action ) {
            $tag = null;

            $template = MRME_DIR . 'templates/admin/tag-edit.php';
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
     * Render the tags list view.
     *
     * @param array|null $notice Optional notice to display.
     * @return void
     */
    private function render_list( ?array $notice ): void {
        global $wpdb;

        $table        = $wpdb->prefix . 'mrme_tags';
        $pivot_table  = $wpdb->prefix . 'mrme_contact_tags';

        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $tags = $wpdb->get_results(
            "SELECT t.*, COUNT(ct.contact_id) AS contact_count
             FROM {$table} t
             LEFT JOIN {$pivot_table} ct ON ct.tag_id = t.id
             GROUP BY t.id
             ORDER BY t.name ASC",
            ARRAY_A
        );

        if ( null === $tags ) {
            $tags = [];
        }

        $template = MRME_DIR . 'templates/admin/tags-list.php';
        if ( file_exists( $template ) ) {
            include $template;
            return;
        }

        // Fallback inline output.
        echo '<div class="wrap">';
        echo '<h1 class="wp-heading-inline">' . esc_html__( 'Tags', 'myrock-mail-engine' ) . '</h1>';
        echo '<a href="' . esc_url( admin_url( 'admin.php?page=mrme-tags&action=new' ) ) . '" class="page-title-action">' . esc_html__( 'Add New', 'myrock-mail-engine' ) . '</a>';

        if ( $notice ) {
            printf( '<div class="notice notice-%s is-dismissible"><p>%s</p></div>', esc_attr( $notice['type'] ), esc_html( $notice['message'] ) );
        }

        echo '<table class="wp-list-table widefat fixed striped"><thead><tr>';
        echo '<th>' . esc_html__( 'Name', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Slug', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Color', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Contacts', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Actions', 'myrock-mail-engine' ) . '</th>';
        echo '</tr></thead><tbody>';

        foreach ( $tags as $row ) {
            $edit_url   = admin_url( 'admin.php?page=mrme-tags&action=edit&id=' . (int) $row['id'] );
            $delete_url = wp_nonce_url(
                admin_url( 'admin-post.php?action=mrme_delete_tag&id=' . (int) $row['id'] ),
                'mrme_delete_tag_' . (int) $row['id']
            );
            $color = ! empty( $row['color'] ) ? $row['color'] : '#cccccc';

            echo '<tr>';
            printf( '<td><a href="%s">%s</a></td>', esc_url( $edit_url ), esc_html( $row['name'] ) );
            echo '<td>' . esc_html( $row['slug'] ) . '</td>';
            printf(
                '<td><span style="display:inline-block;width:16px;height:16px;border-radius:50%%;background:%s;vertical-align:middle;margin-right:5px;"></span>%s</td>',
                esc_attr( $color ),
                esc_html( $color )
            );
            echo '<td>' . esc_html( $row['contact_count'] ) . '</td>';
            printf(
                '<td><a href="%s">%s</a> | <a href="%s" onclick="return confirm(\'%s\')">%s</a></td>',
                esc_url( $edit_url ),
                esc_html__( 'Edit', 'myrock-mail-engine' ),
                esc_url( $delete_url ),
                esc_js( __( 'Are you sure you want to delete this tag?', 'myrock-mail-engine' ) ),
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
     * @param Tag|object|array|null $tag    Existing tag, or null for a new tag.
     * @param array|null            $notice Optional notice to display.
     * @return void
     */
    private function render_edit_form( $tag, ?array $notice ): void {
        $is_new     = null === $tag;
        $page_title = $is_new ? __( 'Add New Tag', 'myrock-mail-engine' ) : __( 'Edit Tag', 'myrock-mail-engine' );

        // Support both object and array representations.
        $tag_id    = $is_new ? 0 : ( is_array( $tag ) ? (int) $tag['id'] : (int) $tag->id );
        $tag_name  = $is_new ? '' : ( is_array( $tag ) ? $tag['name'] : $tag->name );
        $tag_slug  = $is_new ? '' : ( is_array( $tag ) ? $tag['slug'] : $tag->slug );
        $tag_color = $is_new ? '#3182ce' : ( is_array( $tag ) ? $tag['color'] : $tag->color );

        echo '<div class="wrap">';
        echo '<h1>' . esc_html( $page_title ) . '</h1>';

        if ( $notice ) {
            printf( '<div class="notice notice-%s is-dismissible"><p>%s</p></div>', esc_attr( $notice['type'] ), esc_html( $notice['message'] ) );
        }

        echo '<form method="post" action="' . esc_url( admin_url( 'admin-post.php' ) ) . '">';
        echo '<input type="hidden" name="action" value="mrme_save_tag">';
        printf( '<input type="hidden" name="tag_id" value="%d">', $tag_id );
        wp_nonce_field( 'mrme_save_tag' );

        echo '<table class="form-table"><tbody>';

        // Name.
        printf(
            '<tr><th><label for="tag_name">%s</label></th><td><input type="text" id="tag_name" name="name" value="%s" class="regular-text" required></td></tr>',
            esc_html__( 'Name', 'myrock-mail-engine' ),
            esc_attr( $tag_name )
        );

        // Slug.
        printf(
            '<tr><th><label for="tag_slug">%s</label></th><td><input type="text" id="tag_slug" name="slug" value="%s" class="regular-text"><p class="description">%s</p></td></tr>',
            esc_html__( 'Slug', 'myrock-mail-engine' ),
            esc_attr( $tag_slug ),
            esc_html__( 'Leave blank to auto-generate from name.', 'myrock-mail-engine' )
        );

        // Color.
        printf(
            '<tr><th><label for="tag_color">%s</label></th><td><input type="color" id="tag_color" name="color" value="%s"></td></tr>',
            esc_html__( 'Color', 'myrock-mail-engine' ),
            esc_attr( $tag_color )
        );

        echo '</tbody></table>';

        submit_button( $is_new ? __( 'Create Tag', 'myrock-mail-engine' ) : __( 'Update Tag', 'myrock-mail-engine' ) );

        echo '</form>';
        echo '</div>';
    }

    /**
     * Handle saving a tag (create or update).
     *
     * @return void
     */
    public function handle_save(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to perform this action.', 'myrock-mail-engine' ) );
        }

        check_admin_referer( 'mrme_save_tag' );

        $tag_id = isset( $_POST['tag_id'] ) ? (int) $_POST['tag_id'] : 0;
        $name   = isset( $_POST['name'] ) ? sanitize_text_field( wp_unslash( $_POST['name'] ) ) : '';
        $slug   = isset( $_POST['slug'] ) ? sanitize_title( wp_unslash( $_POST['slug'] ) ) : '';
        $color  = isset( $_POST['color'] ) ? sanitize_hex_color( wp_unslash( $_POST['color'] ) ) : '#3182ce';

        $redirect = admin_url( 'admin.php?page=mrme-tags' );

        if ( ! $name ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'name_required', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        // Generate slug from name if not provided.
        if ( ! $slug ) {
            $slug = sanitize_title( $name );
        }

        // Fallback color.
        if ( ! $color ) {
            $color = '#3182ce';
        }

        $data = [
            'name'  => $name,
            'slug'  => $slug,
            'color' => $color,
        ];

        if ( $tag_id > 0 ) {
            $result = Tag::update( $tag_id, $data );
            $action = 'updated';
        } else {
            $result = Tag::create( $data );
            $action = 'created';

            if ( $result ) {
                $tag_id = (int) $result;
            }
        }

        if ( false === $result || is_wp_error( $result ) ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'save_failed', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        $redirect = admin_url( 'admin.php?page=mrme-tags&action=edit&id=' . $tag_id );
        wp_safe_redirect( add_query_arg( [ 'notice' => $action, 'notice_type' => 'success' ], $redirect ) );
        die();
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
            'created'       => __( 'Tag created successfully.', 'myrock-mail-engine' ),
            'updated'       => __( 'Tag updated successfully.', 'myrock-mail-engine' ),
            'deleted'       => __( 'Tag deleted successfully.', 'myrock-mail-engine' ),
            'save_failed'   => __( 'Failed to save tag. Please try again.', 'myrock-mail-engine' ),
            'delete_failed' => __( 'Failed to delete tag. Please try again.', 'myrock-mail-engine' ),
            'name_required' => __( 'Tag name is required.', 'myrock-mail-engine' ),
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
