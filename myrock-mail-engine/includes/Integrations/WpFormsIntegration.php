<?php
/**
 * WPForms integration — captura leads y los guarda como contactos en MRME.
 *
 * Compatible con WPForms Lite y Pro.
 * Se activa solo cuando WPForms está instalado y activo.
 *
 * @package MyRock\MailEngine\Integrations
 */

namespace MyRock\MailEngine\Integrations;

defined( 'ABSPATH' ) || exit;

use MyRock\MailEngine\Services\ContactService;

class WpFormsIntegration {

	/**
	 * Procesa un envío de WPForms y guarda el lead en MRME.
	 *
	 * @param array $fields    Campos enviados, indexados por field_id.
	 * @param array $entry     Datos del entry (no guardado en Lite).
	 * @param array $form_data Configuración del formulario.
	 * @param int   $entry_id  ID del entry guardado (0 en Lite).
	 */
	public function handle_submission( array $fields, array $entry, array $form_data, int $entry_id ): void {
		$form_id = (int) ( $form_data['id'] ?? 0 );

		// ── Verificar si esta integración está habilitada ────────────────────
		$settings = $this->get_settings();

		if ( empty( $settings['enabled'] ) ) {
			return;
		}

		// Si hay formularios específicos configurados, verificar que este esté incluido.
		if ( ! empty( $settings['form_ids'] ) ) {
			$allowed = array_map( 'intval', (array) $settings['form_ids'] );
			if ( ! in_array( $form_id, $allowed, true ) ) {
				return;
			}
		}

		// ── Extraer campos del formulario ────────────────────────────────────
		$contact_data = $this->extract_contact_data( $fields );

		if ( empty( $contact_data['email'] ) || ! is_email( $contact_data['email'] ) ) {
			return; // Sin email válido no hay contacto.
		}

		// ── Asignar lista y tag configurados ─────────────────────────────────
		if ( ! empty( $settings['list_id'] ) ) {
			$contact_data['list_ids'] = [ (int) $settings['list_id'] ];
		}

		if ( ! empty( $settings['tag_id'] ) ) {
			$contact_data['tag_ids'] = [ (int) $settings['tag_id'] ];
		}

		$contact_data['source'] = 'wpforms';
		$contact_data['status'] = 'subscribed';

		// ── Guardar en MRME ──────────────────────────────────────────────────
		$contact_id = ContactService::create( $contact_data );

		if ( $contact_id ) {
			/**
			 * Fires after a WPForms lead is saved to MRME.
			 *
			 * @param int   $contact_id  MRME contact ID.
			 * @param array $contact_data Contact data that was saved.
			 * @param int   $form_id      WPForms form ID.
			 * @param int   $entry_id     WPForms entry ID (0 in Lite).
			 */
			do_action( 'mrme_wpforms_lead_saved', $contact_id, $contact_data, $form_id, $entry_id );
		}
	}

	/**
	 * Recorre los campos de WPForms y extrae email, nombre, teléfono y empresa.
	 *
	 * @param  array $fields Campos del envío.
	 * @return array         Datos del contacto (claves de Contact model).
	 */
	private function extract_contact_data( array $fields ): array {
		$data = [];

		foreach ( $fields as $field ) {
			$type  = (string) ( $field['type']  ?? '' );
			$value = (string) ( $field['value'] ?? '' );

			if ( '' === trim( $value ) ) {
				continue;
			}

			switch ( $type ) {

				case 'email':
					if ( empty( $data['email'] ) ) {
						$data['email'] = sanitize_email( $value );
					}
					break;

				case 'name':
					// WPForms separa first/last en subcampos.
					$first = sanitize_text_field( (string) ( $field['first'] ?? '' ) );
					$last  = sanitize_text_field( (string) ( $field['last']  ?? '' ) );

					if ( $first && empty( $data['first_name'] ) ) {
						$data['first_name'] = $first;
					}
					if ( $last && empty( $data['last_name'] ) ) {
						$data['last_name'] = $last;
					}

					// Fallback: nombre completo en un solo campo texto.
					if ( ! $first && ! $last && empty( $data['first_name'] ) ) {
						$parts = explode( ' ', trim( $value ), 2 );
						$data['first_name'] = sanitize_text_field( $parts[0] );
						if ( isset( $parts[1] ) ) {
							$data['last_name'] = sanitize_text_field( $parts[1] );
						}
					}
					break;

				case 'phone':
					if ( empty( $data['phone'] ) ) {
						$data['phone'] = sanitize_text_field( $value );
					}
					break;

				case 'text':
				case 'textarea':
					// Detectar por label si es empresa/compañía.
					$label = strtolower( (string) ( $field['name'] ?? '' ) );
					if ( empty( $data['company'] ) && $this->label_is_company( $label ) ) {
						$data['company'] = sanitize_text_field( $value );
					}
					break;
			}
		}

		return $data;
	}

	/**
	 * Heurística para detectar si un label de campo corresponde a empresa.
	 */
	private function label_is_company( string $label ): bool {
		$keywords = [ 'empresa', 'company', 'compañía', 'organización', 'organization', 'negocio', 'business' ];
		foreach ( $keywords as $kw ) {
			if ( str_contains( $label, $kw ) ) {
				return true;
			}
		}
		return false;
	}

	/**
	 * Lee la configuración de esta integración desde wp_options.
	 *
	 * @return array{
	 *   enabled:  bool,
	 *   list_id:  int,
	 *   tag_id:   int,
	 *   form_ids: int[],
	 * }
	 */
	public static function get_settings(): array {
		return wp_parse_args(
			(array) get_option( 'mrme_wpforms_settings', [] ),
			[
				'enabled'  => false,
				'list_id'  => 0,
				'tag_id'   => 0,
				'form_ids' => [],
			]
		);
	}

	/**
	 * Guarda la configuración de esta integración.
	 *
	 * @param array $settings
	 */
	public static function save_settings( array $settings ): void {
		update_option( 'mrme_wpforms_settings', [
			'enabled'  => ! empty( $settings['enabled'] ),
			'list_id'  => (int) ( $settings['list_id'] ?? 0 ),
			'tag_id'   => (int) ( $settings['tag_id']  ?? 0 ),
			'form_ids' => array_filter( array_map( 'intval', (array) ( $settings['form_ids'] ?? [] ) ) ),
		] );
	}
}
