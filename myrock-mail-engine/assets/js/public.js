/**
 * MyRock Mail Engine — Public / Frontend JavaScript
 *
 * Handles:
 *  - Client-side validation of [mrme_form] subscription forms before submit.
 *  - Optional AJAX submission when data-ajax="true" is present on the <form>.
 *  - Display of inline success / error messages.
 *
 * Loaded with jQuery as a dependency (wp_enqueue_script dep: ['jquery']).
 * All text strings are provided via wp_localize_script under the
 * `mrmePublic` global object.
 *
 * Usage in shortcode output:
 *   Regular POST (default):  <form class="mrme-form">
 *   AJAX mode:               <form class="mrme-form" data-ajax="true">
 */

/* global mrmePublic, jQuery */

( function ( $ ) {
    'use strict';

    /* ------------------------------------------------------------------ */
    /* Defaults (merged with mrmePublic from wp_localize_script)           */
    /* ------------------------------------------------------------------ */

    var settings = $.extend(
        {
            ajaxurl:  '',
            posturl:  '',
            i18n: {
                required:      'This field is required.',
                invalid_email: 'Please enter a valid email address.',
                success:       'Thank you for subscribing!',
                error:         'An error occurred. Please try again.'
            }
        },
        ( typeof mrmePublic !== 'undefined' ) ? mrmePublic : {}
    );

    /* ------------------------------------------------------------------ */
    /* Utility: show a notice inside a form wrapper                        */
    /* ------------------------------------------------------------------ */

    /**
     * Display a feedback notice (success or error) inside the form wrapper.
     *
     * @param {jQuery}  $form   The <form> element.
     * @param {string}  msg     Notice text.
     * @param {string}  type    'success' | 'error'
     */
    function showNotice( $form, msg, type ) {
        var $wrapper = $form.closest( '.mrme-form-wrapper' );

        // Re-use existing notice element or create one.
        var $notice = $wrapper.find( '.mrme-form__notice' );
        if ( ! $notice.length ) {
            $notice = $( '<div class="mrme-form__notice" role="alert"></div>' );
            $form.before( $notice );
        }

        $notice
            .removeClass( 'mrme-form--success mrme-form--error is-visible' )
            .addClass( 'mrme-form--' + type + ' is-visible' )
            .text( msg );

        // Scroll notice into view if off-screen.
        if ( $notice.offset() ) {
            var noticeTop = $notice.offset().top - 80;
            if ( $( window ).scrollTop() > noticeTop ) {
                $( 'html, body' ).animate( { scrollTop: noticeTop }, 300 );
            }
        }
    }

    /**
     * Hide the notice.
     *
     * @param {jQuery} $form
     */
    function hideNotice( $form ) {
        $form.closest( '.mrme-form-wrapper' ).find( '.mrme-form__notice' ).removeClass( 'is-visible' );
    }

    /* ------------------------------------------------------------------ */
    /* Field validation helpers                                             */
    /* ------------------------------------------------------------------ */

    /**
     * Mark a field as invalid with an error message.
     *
     * @param {jQuery} $field  The <input> element.
     * @param {string} msg     Error message.
     */
    function setFieldError( $field, msg ) {
        $field
            .addClass( 'is-invalid' )
            .attr( 'aria-invalid', 'true' )
            .removeClass( 'is-valid' );

        var $wrap = $field.closest( '.mrme-form__field' );
        $wrap.addClass( 'has-error' );

        var $errMsg = $wrap.find( '.mrme-form__error-msg' );
        if ( ! $errMsg.length ) {
            $errMsg = $( '<span class="mrme-form__error-msg"></span>' );
            $field.after( $errMsg );
        }

        $errMsg.text( msg );
    }

    /**
     * Clear a field's error state.
     *
     * @param {jQuery} $field
     */
    function clearFieldError( $field ) {
        $field
            .removeClass( 'is-invalid' )
            .attr( 'aria-invalid', 'false' )
            .addClass( 'is-valid' );

        var $wrap = $field.closest( '.mrme-form__field' );
        $wrap.removeClass( 'has-error' );
        $wrap.find( '.mrme-form__error-msg' ).text( '' );
    }

    /**
     * Validate a single input field.
     *
     * @param {jQuery} $field
     * @returns {boolean} True if valid.
     */
    function validateField( $field ) {
        var val      = $field.val().trim();
        var type     = $field.attr( 'type' ) || 'text';
        var required = $field.prop( 'required' ) || $field.attr( 'aria-required' ) === 'true';

        // Required check.
        if ( required && val === '' ) {
            setFieldError( $field, settings.i18n.required );
            return false;
        }

        // Email format check.
        if ( type === 'email' && val !== '' ) {
            var emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;
            if ( ! emailRe.test( val ) ) {
                setFieldError( $field, settings.i18n.invalid_email );
                return false;
            }
        }

        clearFieldError( $field );
        return true;
    }

    /**
     * Validate all inputs in a form.
     *
     * @param {jQuery} $form
     * @returns {boolean} True if all fields are valid.
     */
    function validateForm( $form ) {
        var valid = true;
        var $firstInvalid = null;

        $form.find( '.mrme-form__input' ).each( function () {
            var $field = $( this );
            // Skip honeypot fields.
            if ( $field.closest( '.mrme-form__field--honeypot' ).length ) {
                return;
            }
            if ( ! validateField( $field ) ) {
                valid = false;
                if ( ! $firstInvalid ) {
                    $firstInvalid = $field;
                }
            }
        } );

        if ( $firstInvalid ) {
            $firstInvalid.trigger( 'focus' );
        }

        return valid;
    }

    /* ------------------------------------------------------------------ */
    /* Live validation on blur                                              */
    /* ------------------------------------------------------------------ */

    $( document ).on( 'blur', '.mrme-form .mrme-form__input', function () {
        validateField( $( this ) );
    } );

    // Clear error state on input.
    $( document ).on( 'input', '.mrme-form .mrme-form__input.is-invalid', function () {
        clearFieldError( $( this ) );
    } );

    /* ------------------------------------------------------------------ */
    /* Form submit handler                                                  */
    /* ------------------------------------------------------------------ */

    $( document ).on( 'submit', '.mrme-form', function ( e ) {
        var $form    = $( this );
        var isAjax   = $form.data( 'ajax' ) === true || $form.attr( 'data-ajax' ) === 'true';
        var $submit  = $form.find( '.mrme-form__submit' );

        // Always validate first.
        if ( ! validateForm( $form ) ) {
            e.preventDefault();
            return;
        }

        // If not AJAX mode, let the browser do a regular POST.
        if ( ! isAjax ) {
            // Show the button in loading state while the page navigates.
            $submit.addClass( 'is-loading' ).prop( 'disabled', true );
            return;
        }

        // ----------------------------------------------------------------
        // AJAX mode
        // ----------------------------------------------------------------
        e.preventDefault();

        hideNotice( $form );
        $submit.addClass( 'is-loading' ).prop( 'disabled', true );

        var formData = $form.serialize();

        $.post(
            settings.posturl || $form.attr( 'action' ),
            formData + '&mrme_ajax=1',
            function ( response ) {
                if ( response && response.success ) {
                    // Replace the form with a success message.
                    var successMsg = ( response.data && response.data.message )
                        ? response.data.message
                        : settings.i18n.success;

                    $form.addClass( 'is-submitted' );

                    var $successEl = $form.closest( '.mrme-form-wrapper' ).find( '.mrme-form-success-message' );
                    if ( ! $successEl.length ) {
                        $successEl = $( '<div class="mrme-form-success-message" role="status"></div>' );
                        $form.after( $successEl );
                    }

                    $successEl.text( successMsg ).show();

                    // Fire an event so themes / plugins can respond.
                    $form.trigger( 'mrme:subscribed', [ response.data ] );

                } else {
                    var errMsg = ( response && response.data && response.data.message )
                        ? response.data.message
                        : settings.i18n.error;

                    showNotice( $form, errMsg, 'error' );
                    $submit.removeClass( 'is-loading' ).prop( 'disabled', false );
                }
            },
            'json'
        ).fail( function ( jqXHR ) {
            var msg = settings.i18n.error;

            // If the server returned a JSON error body, try to read it.
            if ( jqXHR.responseJSON && jqXHR.responseJSON.message ) {
                msg = jqXHR.responseJSON.message;
            }

            showNotice( $form, msg, 'error' );
            $submit.removeClass( 'is-loading' ).prop( 'disabled', false );
        } );
    } );

    /* ------------------------------------------------------------------ */
    /* Auto-dismiss page-reload notices                                    */
    /*                                                                     */
    /* When submitted via regular POST, the page reloads with ?mrme=…     */
    /* The shortcode already outputs the notice, but we animate it in and  */
    /* optionally auto-dismiss it after a delay.                           */
    /* ------------------------------------------------------------------ */

    ( function initPageReloadNotice() {
        var $notice = $( '.mrme-form__notice' );
        if ( ! $notice.length ) { return; }

        // Animate in.
        $notice.hide().fadeIn( 300 );

        // Auto-dismiss success notices after 8 seconds.
        if ( $notice.hasClass( 'mrme-form--success' ) ) {
            setTimeout( function () {
                $notice.fadeOut( 500 );
            }, 8000 );
        }
    } )();

} )( jQuery );
