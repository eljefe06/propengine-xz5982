/**
 * MyRock Mail Engine — Admin JavaScript
 *
 * Handles:
 *  - Modal open / close (import CSV modal and any generic .mrme-modal)
 *  - Test send AJAX
 *  - SMTP settings toggle (show / hide based on provider radio)
 *  - Automation step builder (add / remove steps)
 *  - Check-all checkbox for bulk tables
 *  - Confirm dialogs for destructive actions
 *  - API key copy-to-clipboard
 *  - JSON meta field validation
 *
 * Requires: jQuery (loaded by WordPress admin).
 */

/* global mrmeAdmin, ajaxurl */

( function ( $ ) {
    'use strict';

    /* ------------------------------------------------------------------ */
    /* Utilities                                                            */
    /* ------------------------------------------------------------------ */

    /**
     * Show an inline notice inside a target element.
     *
     * @param {jQuery}  $el     Target container.
     * @param {string}  msg     Message text.
     * @param {string}  type    'success' | 'error'
     * @param {number}  [ttl]   Auto-hide after N ms (0 = never).
     */
    function showInlineNotice( $el, msg, type, ttl ) {
        $el
            .removeClass( 'mrme-inline-notice--success mrme-inline-notice--error' )
            .addClass( 'mrme-inline-notice--' + type )
            .text( msg )
            .show();

        if ( ttl ) {
            setTimeout( function () {
                $el.fadeOut( 300 );
            }, ttl );
        }
    }

    /* ------------------------------------------------------------------ */
    /* Modal                                                                */
    /* ------------------------------------------------------------------ */

    /**
     * Open a modal by toggling aria-hidden and the .is-open class.
     *
     * @param {string} modalId  The value of the modal element's id attribute.
     */
    function openModal( modalId ) {
        var $modal = $( '#' + modalId );
        if ( ! $modal.length ) { return; }
        $modal.attr( 'aria-hidden', 'false' ).addClass( 'is-open' );
        $( 'body' ).addClass( 'mrme-modal-open' );
        // Trap focus: move focus to first focusable element inside.
        setTimeout( function () {
            $modal.find( 'input, select, textarea, button, [tabindex]' ).not( '[disabled]' ).first().trigger( 'focus' );
        }, 50 );
    }

    /**
     * Close all open modals.
     */
    function closeAllModals() {
        $( '.mrme-modal.is-open, .mrme-modal[aria-hidden="false"]' )
            .attr( 'aria-hidden', 'true' )
            .removeClass( 'is-open' );
        $( 'body' ).removeClass( 'mrme-modal-open' );
    }

    // Open import-CSV modal.
    $( document ).on( 'click', '#mrme-open-import-modal', function ( e ) {
        e.preventDefault();
        openModal( 'mrme-import-modal' );
    } );

    // Close modal via overlay click or [data-modal-close] elements.
    $( document ).on( 'click', '[data-modal-close]', function ( e ) {
        e.preventDefault();
        closeAllModals();
    } );

    // Close modal on Escape key.
    $( document ).on( 'keydown', function ( e ) {
        if ( e.key === 'Escape' || e.keyCode === 27 ) {
            closeAllModals();
        }
    } );

    /* ------------------------------------------------------------------ */
    /* Test send AJAX                                                       */
    /* ------------------------------------------------------------------ */

    $( document ).on( 'click', '#mrme-send-test-btn', function ( e ) {
        e.preventDefault();

        var $btn         = $( this );
        var campaignId   = $btn.data( 'campaign-id' );
        var nonce        = $btn.data( 'nonce' );
        var testEmail    = $( '#mrme-test-email' ).val().trim();
        var $msg         = $( '#mrme-test-send-msg' );

        if ( ! testEmail ) {
            showInlineNotice( $msg, mrmeAdmin.i18n.test_email_required, 'error', 4000 );
            $( '#mrme-test-email' ).trigger( 'focus' );
            return;
        }

        // Very basic email format check.
        var emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if ( ! emailRe.test( testEmail ) ) {
            showInlineNotice( $msg, mrmeAdmin.i18n.invalid_email, 'error', 4000 );
            return;
        }

        $btn.prop( 'disabled', true ).text( mrmeAdmin.i18n.sending );

        $.post(
            ajaxurl,
            {
                action:      'mrme_send_test',
                campaign_id: campaignId,
                test_email:  testEmail,
                _wpnonce:    nonce
            },
            function ( response ) {
                if ( response && response.success ) {
                    showInlineNotice( $msg, response.data.message || mrmeAdmin.i18n.test_sent, 'success', 5000 );
                } else {
                    var errMsg = ( response && response.data && response.data.message )
                        ? response.data.message
                        : mrmeAdmin.i18n.test_failed;
                    showInlineNotice( $msg, errMsg, 'error', 6000 );
                }
            }
        ).fail( function () {
            showInlineNotice( $msg, mrmeAdmin.i18n.ajax_error, 'error', 6000 );
        } ).always( function () {
            $btn.prop( 'disabled', false ).text( mrmeAdmin.i18n.send_test );
        } );
    } );

    /* ------------------------------------------------------------------ */
    /* SMTP settings toggle                                                 */
    /* ------------------------------------------------------------------ */

    function toggleProviderSections() {
        var selected = $( '.mrme-provider-radio:checked' ).val();
        if ( 'smtp' === selected ) {
            $( '#mrme-smtp-settings' ).slideDown( 200 );
        } else {
            $( '#mrme-smtp-settings' ).slideUp( 200 );
        }
        if ( 'mailgun' === selected ) {
            $( '#mrme-mailgun-settings' ).slideDown( 200 );
        } else {
            $( '#mrme-mailgun-settings' ).slideUp( 200 );
        }
    }

    $( document ).on( 'change', '.mrme-provider-radio', toggleProviderSections );

    // Run on load.
    toggleProviderSections();

    /* ------------------------------------------------------------------ */
    /* Test Mailgun connection (settings page)                              */
    /* ------------------------------------------------------------------ */

    $( document ).on( 'click', '#mrme-test-mailgun-btn', function ( e ) {
        e.preventDefault();

        var $btn    = $( this );
        var nonce   = $btn.data( 'nonce' );
        var email   = $btn.data( 'email' );
        var $result = $( '#mrme-mailgun-test-result' );

        $btn.prop( 'disabled', true );

        $.post(
            ajaxurl,
            {
                action:   'mrme_test_mailgun',
                _wpnonce: nonce,
                email:    email,
                api_key:  $( '#mrme-mg-api-key' ).val(),
                domain:   $( '#mrme-mg-domain' ).val(),
                region:   $( '#mrme-mg-region' ).val()
            },
            function ( response ) {
                if ( response && response.success ) {
                    showInlineNotice( $result, response.data.message || 'Mailgun OK!', 'success', 6000 );
                } else {
                    var errMsg = ( response && response.data && response.data.message )
                        ? response.data.message
                        : 'Mailgun test failed.';
                    showInlineNotice( $result, errMsg, 'error', 8000 );
                }
            }
        ).fail( function () {
            showInlineNotice( $result, mrmeAdmin.i18n.ajax_error, 'error', 6000 );
        } ).always( function () {
            $btn.prop( 'disabled', false );
        } );
    } );

    /* ------------------------------------------------------------------ */
    /* Test SMTP connection (settings page)                                 */
    /* ------------------------------------------------------------------ */

    $( document ).on( 'click', '#mrme-test-smtp-btn', function ( e ) {
        e.preventDefault();

        var $btn    = $( this );
        var nonce   = $btn.data( 'nonce' );
        var $result = $( '#mrme-smtp-test-result' );

        $btn.prop( 'disabled', true );

        $.post(
            ajaxurl,
            {
                action:   'mrme_test_smtp',
                _wpnonce: nonce,
                host:     $( '#mrme-smtp-host' ).val(),
                port:     $( '#mrme-smtp-port' ).val(),
                enc:      $( '#mrme-smtp-encryption' ).val(),
                user:     $( '#mrme-smtp-username' ).val(),
                pass:     $( '#mrme-smtp-password' ).val()
            },
            function ( response ) {
                if ( response && response.success ) {
                    showInlineNotice( $result, response.data.message || mrmeAdmin.i18n.smtp_ok, 'success', 6000 );
                } else {
                    var errMsg = ( response && response.data && response.data.message )
                        ? response.data.message
                        : mrmeAdmin.i18n.smtp_fail;
                    showInlineNotice( $result, errMsg, 'error', 8000 );
                }
            }
        ).fail( function () {
            showInlineNotice( $result, mrmeAdmin.i18n.ajax_error, 'error', 6000 );
        } ).always( function () {
            $btn.prop( 'disabled', false );
        } );
    } );

    /* ------------------------------------------------------------------ */
    /* Automation step builder                                              */
    /* ------------------------------------------------------------------ */

    var stepCount = $( '.mrme-step-row' ).length;

    /**
     * Build a new step row HTML string.
     *
     * @param {number} index  Row index for naming inputs.
     * @returns {string} HTML.
     */
    function buildStepRow( index ) {
        return '<div class="mrme-step-row" data-step="' + index + '">' +
            '<span class="mrme-step-row__handle dashicons dashicons-menu" title="Drag to reorder"></span>' +

            '<label>' +
                '<span class="screen-reader-text">Step type</span>' +
                '<select name="steps[' + index + '][type]">' +
                    '<option value="wait">' + mrmeAdmin.i18n.step_wait + '</option>' +
                    '<option value="send_email">' + mrmeAdmin.i18n.step_send_email + '</option>' +
                    '<option value="add_tag">' + mrmeAdmin.i18n.step_add_tag + '</option>' +
                    '<option value="remove_tag">' + mrmeAdmin.i18n.step_remove_tag + '</option>' +
                    '<option value="change_status">' + mrmeAdmin.i18n.step_change_status + '</option>' +
                '</select>' +
            '</label>' +

            '<label>' +
                '<span class="screen-reader-text">Delay amount</span>' +
                '<input type="number" name="steps[' + index + '][delay_amount]" value="1" min="0" style="width:60px">' +
            '</label>' +

            '<label>' +
                '<span class="screen-reader-text">Delay unit</span>' +
                '<select name="steps[' + index + '][delay_unit]">' +
                    '<option value="minutes">' + mrmeAdmin.i18n.delay_minutes + '</option>' +
                    '<option value="hours">' + mrmeAdmin.i18n.delay_hours + '</option>' +
                    '<option value="days" selected>' + mrmeAdmin.i18n.delay_days + '</option>' +
                    '<option value="weeks">' + mrmeAdmin.i18n.delay_weeks + '</option>' +
                '</select>' +
            '</label>' +

            '<button type="button" class="mrme-step-row__remove dashicons dashicons-trash" aria-label="' + mrmeAdmin.i18n.remove_step + '"></button>' +
        '</div>';
    }

    $( document ).on( 'click', '#mrme-add-step', function () {
        stepCount++;
        $( '.mrme-automation-steps' ).append( buildStepRow( stepCount ) );
    } );

    $( document ).on( 'click', '.mrme-step-row__remove', function () {
        $( this ).closest( '.mrme-step-row' ).remove();
    } );

    /* ------------------------------------------------------------------ */
    /* Check-all checkbox                                                   */
    /* ------------------------------------------------------------------ */

    $( document ).on( 'change', '#mrme-check-all', function () {
        var checked = $( this ).prop( 'checked' );
        $( this ).closest( 'form' ).find( 'input[type="checkbox"][name$="[]"]' ).prop( 'checked', checked );
    } );

    /* ------------------------------------------------------------------ */
    /* Confirm dialogs for destructive actions                              */
    /* ------------------------------------------------------------------ */

    $( document ).on( 'click', '[data-confirm]', function ( e ) {
        var msg = $( this ).data( 'confirm' );
        if ( msg && ! window.confirm( msg ) ) {
            e.preventDefault();
        }
    } );

    // Extra safety for send buttons on campaigns list.
    $( document ).on( 'click', '.mrme-action--send', function ( e ) {
        var msg = $( this ).data( 'confirm' ) || mrmeAdmin.i18n.confirm_send;
        if ( ! window.confirm( msg ) ) {
            e.preventDefault();
        }
    } );

    // Delete links that aren't inside a data-confirm already get a default message.
    $( document ).on( 'click', '.mrme-action--delete:not([data-confirm])', function ( e ) {
        if ( ! window.confirm( mrmeAdmin.i18n.confirm_delete ) ) {
            e.preventDefault();
        }
    } );

    /* ------------------------------------------------------------------ */
    /* API key copy-to-clipboard                                            */
    /* ------------------------------------------------------------------ */

    $( document ).on( 'click', '#mrme-copy-api-key', function () {
        var $btn    = $( this );
        var target  = $btn.data( 'clipboard-target' );
        var text    = $( target ).text().trim();

        if ( navigator.clipboard ) {
            navigator.clipboard.writeText( text ).then( function () {
                $btn.text( mrmeAdmin.i18n.copied );
                setTimeout( function () { $btn.text( mrmeAdmin.i18n.copy ); }, 2000 );
            } ).catch( function () {
                fallbackCopy( text );
            } );
        } else {
            fallbackCopy( text );
        }
    } );

    function fallbackCopy( text ) {
        var $tmp = $( '<textarea>' ).val( text ).css( { position: 'absolute', left: '-9999px' } );
        $( 'body' ).append( $tmp );
        $tmp[0].select();
        try {
            document.execCommand( 'copy' );
        } catch ( err ) {
            // Silent fallback.
        }
        $tmp.remove();
    }

    /* ------------------------------------------------------------------ */
    /* JSON meta field validation                                           */
    /* ------------------------------------------------------------------ */

    $( document ).on( 'blur', '.mrme-meta-textarea', function () {
        var val = $( this ).val().trim();
        var $err = $( '#mrme-meta-error' );

        if ( ! val || val === '{}' || val === '[]' ) {
            $err.hide();
            return;
        }

        try {
            JSON.parse( val );
            $err.hide();
        } catch ( e ) {
            $err.show();
        }
    } );

    // Prevent form submit when JSON is invalid.
    $( document ).on( 'submit', '.mrme-edit-form', function ( e ) {
        var $metaField = $( this ).find( '.mrme-meta-textarea' );
        if ( ! $metaField.length ) { return; }

        var val = $metaField.val().trim();
        if ( val && val !== '{}' && val !== '[]' ) {
            try {
                JSON.parse( val );
            } catch ( err ) {
                e.preventDefault();
                $( '#mrme-meta-error' ).show();
                $metaField.trigger( 'focus' );
            }
        }
    } );

    /* ------------------------------------------------------------------ */
    /* i18n defaults (merged with mrmeAdmin.i18n from wp_localize_script)  */
    /* ------------------------------------------------------------------ */

    // Ensure mrmeAdmin is available (set by the enqueue in AdminAssets).
    if ( typeof mrmeAdmin === 'undefined' ) {
        window.mrmeAdmin = {};
    }

    mrmeAdmin.i18n = $.extend(
        {
            test_email_required: 'Please enter a test email address.',
            invalid_email:       'Please enter a valid email address.',
            sending:             'Sending…',
            send_test:           'Send Test',
            test_sent:           'Test email sent successfully!',
            test_failed:         'Could not send test email.',
            ajax_error:          'A network error occurred. Please try again.',
            smtp_ok:             'SMTP connection successful!',
            smtp_fail:           'SMTP connection failed.',
            confirm_delete:      'Are you sure you want to delete this? This action cannot be undone.',
            confirm_send:        'Send this campaign to all recipients now?',
            copied:              'Copied!',
            copy:                'Copy',
            step_wait:           'Wait',
            step_send_email:     'Send Email',
            step_add_tag:        'Add Tag',
            step_remove_tag:     'Remove Tag',
            step_change_status:  'Change Status',
            delay_minutes:       'Minutes',
            delay_hours:         'Hours',
            delay_days:          'Days',
            delay_weeks:         'Weeks',
            remove_step:         'Remove step'
        },
        mrmeAdmin.i18n || {}
    );

} )( jQuery );
