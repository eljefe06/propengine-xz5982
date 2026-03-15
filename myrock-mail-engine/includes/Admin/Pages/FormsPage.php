<?php
namespace MyRock\MailEngine\Admin\Pages;

defined('ABSPATH') || exit;


use MyRock\MailEngine\Models\Form;

/**
 * Admin Forms page for MyRock Mail Engine.
 */
class FormsPage {

    /**
     * Render the forms page.
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
            $form = Form::find( $id );

            if ( ! $form ) {
                wp_die( esc_html__( 'Form not found.', 'myrock-mail-engine' ) );
            }

            $lists = $this->get_all_lists();
            $tags  = $this->get_all_tags();

            $template = MRME_DIR . 'templates/admin/form-edit.php';
            if ( file_exists( $template ) ) {
                include $template;
            } else {
                $this->render_edit_form( $form, $lists, $tags, $notice );
            }
            return;
        }

        if ( 'new' === $action ) {
            $form  = null;
            $lists = $this->get_all_lists();
            $tags  = $this->get_all_tags();

            $template = MRME_DIR . 'templates/admin/form-edit.php';
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
     * Render the forms list view.
     *
     * @param array|null $notice Optional notice to display.
     * @return void
     */
    private function render_list( ?array $notice ): void {
        global $wpdb;

        $table = $wpdb->prefix . 'mrme_forms';

        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $forms = $wpdb->get_results(
            "SELECT * FROM {$table} ORDER BY created_at DESC",
            ARRAY_A
        );

        if ( null === $forms ) {
            $forms = [];
        }

        $template = MRME_DIR . 'templates/admin/forms-list.php';
        if ( file_exists( $template ) ) {
            include $template;
            return;
        }

        // Fallback inline output.
        echo '<div class="wrap">';
        echo '<h1 class="wp-heading-inline">' . esc_html__( 'Signup Forms', 'myrock-mail-engine' ) . '</h1>';
        echo '<a href="' . esc_url( admin_url( 'admin.php?page=mrme-forms&action=new' ) ) . '" class="page-title-action">' . esc_html__( 'Add New', 'myrock-mail-engine' ) . '</a>';

        if ( $notice ) {
            printf( '<div class="notice notice-%s is-dismissible"><p>%s</p></div>', esc_attr( $notice['type'] ), esc_html( $notice['message'] ) );
        }

        echo '<table class="wp-list-table widefat fixed striped"><thead><tr>';
        echo '<th>' . esc_html__( 'Name', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Status', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Double Opt-In', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Shortcode', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Actions', 'myrock-mail-engine' ) . '</th>';
        echo '</tr></thead><tbody>';

        foreach ( $forms as $row ) {
            $edit_url   = admin_url( 'admin.php?page=mrme-forms&action=edit&id=' . (int) $row['id'] );
            $delete_url = wp_nonce_url(
                admin_url( 'admin-post.php?action=mrme_delete_form&id=' . (int) $row['id'] ),
                'mrme_delete_form_' . (int) $row['id']
            );
            $shortcode = sprintf( '[mrme_form id="%d"]', (int) $row['id'] );

            echo '<tr>';
            printf( '<td><a href="%s">%s</a></td>', esc_url( $edit_url ), esc_html( $row['name'] ) );
            echo '<td>' . esc_html( $row['status'] ) . '</td>';
            echo '<td>' . ( $row['double_optin'] ? esc_html__( 'Yes', 'myrock-mail-engine' ) : esc_html__( 'No', 'myrock-mail-engine' ) ) . '</td>';
            printf( '<td><code>%s</code></td>', esc_html( $shortcode ) );
            printf(
                '<td><a href="%s">%s</a> | <a href="%s" onclick="return confirm(\'%s\')">%s</a></td>',
                esc_url( $edit_url ),
                esc_html__( 'Edit', 'myrock-mail-engine' ),
                esc_url( $delete_url ),
                esc_js( __( 'Are you sure you want to delete this form?', 'myrock-mail-engine' ) ),
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
     * @param Form|object|array|null $form   Existing form, or null for a new one.
     * @param array                  $lists  All available lists.
     * @param array                  $tags   All available tags.
     * @param array|null             $notice Optional notice to display.
     * @return void
     */
    private function render_edit_form( $form, array $lists, array $tags, ?array $notice ): void {
        $is_new     = null === $form;
        $page_title = $is_new ? __( 'Add New Form', 'myrock-mail-engine' ) : __( 'Edit Form', 'myrock-mail-engine' );

        $form_id         = $is_new ? 0 : ( is_array( $form ) ? (int) $form['id'] : (int) $form->id );
        $form_name       = $is_new ? '' : ( is_array( $form ) ? $form['name'] : $form->name );
        $form_fields     = $is_new ? '[]' : ( is_array( $form ) ? $form['fields'] : $form->fields );
        $form_list_ids   = $is_new ? [] : ( is_array( $form ) ? explode( ',', $form['list_ids'] ) : explode( ',', $form->list_ids ) );
        $form_tag_ids    = $is_new ? [] : ( is_array( $form ) ? explode( ',', $form['tag_ids'] ) : explode( ',', $form->tag_ids ) );
        $double_optin    = $is_new ? 0 : ( is_array( $form ) ? (int) $form['double_optin'] : (int) $form->double_optin );
        $success_message = $is_new ? '' : ( is_array( $form ) ? $form['success_message'] : $form->success_message );
        $redirect_url    = $is_new ? '' : ( is_array( $form ) ? $form['redirect_url'] : $form->redirect_url );
        $form_status     = $is_new ? 'active' : ( is_array( $form ) ? $form['status'] : $form->status );

        echo '<div class="wrap">';
        echo '<h1>' . esc_html( $page_title ) . '</h1>';

        if ( $notice ) {
            printf( '<div class="notice notice-%s is-dismissible"><p>%s</p></div>', esc_attr( $notice['type'] ), esc_html( $notice['message'] ) );
        }

        echo '<form method="post" action="' . esc_url( admin_url( 'admin-post.php' ) ) . '">';
        echo '<input type="hidden" name="action" value="mrme_save_form">';
        printf( '<input type="hidden" name="form_id" value="%d">', $form_id );
        wp_nonce_field( 'mrme_save_form' );

        echo '<table class="form-table"><tbody>';

        // Name.
        printf(
            '<tr><th><label for="form_name">%s</label></th><td><input type="text" id="form_name" name="name" value="%s" class="regular-text" required></td></tr>',
            esc_html__( 'Form Name', 'myrock-mail-engine' ),
            esc_attr( $form_name )
        );

        // Fields JSON.
        printf(
            '<tr><th><label for="form_fields">%s</label></th><td><textarea id="form_fields" name="fields" rows="10" class="large-text code">%s</textarea><p class="description">%s</p></td></tr>',
            esc_html__( 'Fields (JSON)', 'myrock-mail-engine' ),
            esc_textarea( $form_fields ),
            esc_html__( 'Define form fields as a JSON array.', 'myrock-mail-engine' )
        );

        // Lists.
        echo '<tr><th>' . esc_html__( 'Lists', 'myrock-mail-engine' ) . '</th><td>';
        foreach ( $lists as $lst ) {
            $lst_id      = is_array( $lst ) ? (int) $lst['id'] : (int) $lst->id;
            $lst_name    = is_array( $lst ) ? $lst['name'] : $lst->name;
            $is_selected = in_array( (string) $lst_id, array_map( 'strval', $form_list_ids ), true );
            printf(
                '<label><input type="checkbox" name="list_ids[]" value="%d" %s> %s</label><br>',
                $lst_id,
                checked( $is_selected, true, false ),
                esc_html( $lst_name )
            );
        }
        echo '</td></tr>';

        // Tags.
        echo '<tr><th>' . esc_html__( 'Tags', 'myrock-mail-engine' ) . '</th><td>';
        foreach ( $tags as $tag ) {
            $tag_id      = is_array( $tag ) ? (int) $tag['id'] : (int) $tag->id;
            $tag_name    = is_array( $tag ) ? $tag['name'] : $tag->name;
            $is_selected = in_array( (string) $tag_id, array_map( 'strval', $form_tag_ids ), true );
            printf(
                '<label><input type="checkbox" name="tag_ids[]" value="%d" %s> %s</label><br>',
                $tag_id,
                checked( $is_selected, true, false ),
                esc_html( $tag_name )
            );
        }
        echo '</td></tr>';

        // Double Opt-In.
        printf(
            '<tr><th>%s</th><td><label><input type="checkbox" name="double_optin" value="1" %s> %s</label></td></tr>',
            esc_html__( 'Double Opt-In', 'myrock-mail-engine' ),
            checked( $double_optin, 1, false ),
            esc_html__( 'Send confirmation email before subscribing', 'myrock-mail-engine' )
        );

        // Success Message.
        printf(
            '<tr><th><label for="success_message">%s</label></th><td><textarea id="success_message" name="success_message" rows="3" class="large-text">%s</textarea></td></tr>',
            esc_html__( 'Success Message', 'myrock-mail-engine' ),
            esc_textarea( $success_message )
        );

        // Redirect URL.
        printf(
            '<tr><th><label for="redirect_url">%s</label></th><td><input type="url" id="redirect_url" name="redirect_url" value="%s" class="regular-text"><p class="description">%s</p></td></tr>',
            esc_html__( 'Redirect URL', 'myrock-mail-engine' ),
            esc_attr( $redirect_url ),
            esc_html__( 'Optional URL to redirect after successful submission.', 'myrock-mail-engine' )
        );

        // Status.
        echo '<tr><th><label for="form_status">' . esc_html__( 'Status', 'myrock-mail-engine' ) . '</label></th><td>';
        echo '<select id="form_status" name="status">';
        $statuses = [ 'active' => __( 'Active', 'myrock-mail-engine' ), 'inactive' => __( 'Inactive', 'myrock-mail-engine' ) ];
        foreach ( $statuses as $val => $label ) {
            printf( '<option value="%s" %s>%s</option>', esc_attr( $val ), selected( $form_status, $val, false ), esc_html( $label ) );
        }
        echo '</select>';
        echo '</td></tr>';

        echo '</tbody></table>';

        submit_button( $is_new ? __( 'Create Form', 'myrock-mail-engine' ) : __( 'Update Form', 'myrock-mail-engine' ) );

        echo '</form>';
        echo '</div>';
    }

    /**
     * Handle saving a form (create or update).
     *
     * @return void
     */
    public function handle_save(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to perform this action.', 'myrock-mail-engine' ) );
        }

        check_admin_referer( 'mrme_save_form' );

        $form_id = isset( $_POST['form_id'] ) ? (int) $_POST['form_id'] : 0;
        $name    = isset( $_POST['name'] ) ? sanitize_text_field( wp_unslash( $_POST['name'] ) ) : '';

        $redirect = admin_url( 'admin.php?page=mrme-forms' );

        if ( ! $name ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'name_required', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        // Sanitize and validate fields JSON.
        $fields_raw = isset( $_POST['fields'] ) ? wp_unslash( $_POST['fields'] ) : '[]'; // phpcs:ignore WordPress.Security.ValidatedSanitizedInput
        $fields_decoded = json_decode( $fields_raw, true );

        if ( null === $fields_decoded || ! is_array( $fields_decoded ) ) {
            $fields_json = '[]';
        } else {
            $fields_json = wp_json_encode( $fields_decoded );
        }

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

        $double_optin    = isset( $_POST['double_optin'] ) ? 1 : 0;
        $success_message = isset( $_POST['success_message'] ) ? sanitize_textarea_field( wp_unslash( $_POST['success_message'] ) ) : '';
        $redirect_url    = isset( $_POST['redirect_url'] ) ? esc_url_raw( wp_unslash( $_POST['redirect_url'] ) ) : '';
        $status          = isset( $_POST['status'] ) ? sanitize_key( $_POST['status'] ) : 'active';

        if ( ! in_array( $status, [ 'active', 'inactive' ], true ) ) {
            $status = 'active';
        }

        $data = [
            'name'            => $name,
            'fields'          => $fields_json,
            'list_ids'        => $list_ids,
            'tag_ids'         => $tag_ids,
            'double_optin'    => $double_optin,
            'success_message' => $success_message,
            'redirect_url'    => $redirect_url,
            'status'          => $status,
        ];

        if ( $form_id > 0 ) {
            $result = Form::update( $form_id, $data );
            $action = 'updated';
        } else {
            $result = Form::create( $data );
            $action = 'created';

            if ( $result ) {
                $form_id = (int) $result;
            }
        }

        if ( false === $result || is_wp_error( $result ) ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'save_failed', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        $redirect = admin_url( 'admin.php?page=mrme-forms&action=edit&id=' . $form_id );
        wp_safe_redirect( add_query_arg( [ 'notice' => $action, 'notice_type' => 'success' ], $redirect ) );
        die();
    }

    /**
     * Fetch all mailing lists for the form selector.
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
     * Fetch all tags for the form selector.
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
            'created'       => __( 'Form created successfully.', 'myrock-mail-engine' ),
            'updated'       => __( 'Form updated successfully.', 'myrock-mail-engine' ),
            'deleted'       => __( 'Form deleted successfully.', 'myrock-mail-engine' ),
            'save_failed'   => __( 'Failed to save form. Please try again.', 'myrock-mail-engine' ),
            'delete_failed' => __( 'Failed to delete form. Please try again.', 'myrock-mail-engine' ),
            'name_required' => __( 'Form name is required.', 'myrock-mail-engine' ),
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
