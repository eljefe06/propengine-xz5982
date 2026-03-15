<?php
/**
 * Admin template: Create / Edit a Signup Form — Visual Builder.
 *
 * Available variables (set by FormsPage before including this template):
 *   @var array|null $form    Form row (null when creating a new form).
 *   @var array      $lists   All available mailing lists.
 *   @var array      $tags    All available tags.
 *   @var array|null $notice  Optional notice ['type', 'message'].
 *
 * @package MyRock\MailEngine
 */

defined( 'ABSPATH' ) || exit;

$is_new   = empty( $form );
$base_url = admin_url( 'admin.php?page=mrme-forms' );

// Helper to read a field from $form (array or object).
$field = static function ( string $key, $default = '' ) use ( $form ) {
	if ( ! $form ) {
		return $default;
	}
	return is_array( $form ) ? ( $form[ $key ] ?? $default ) : ( $form->$key ?? $default );
};

$form_id         = (int) $field( 'id', 0 );
$form_name       = $field( 'name', '' );
$form_status     = $field( 'status', 'active' );
$double_optin    = (int) $field( 'double_optin', 0 );
$success_message = $field( 'success_message', __( '¡Gracias! Te has suscrito correctamente.', 'myrock-mail-engine' ) );
$redirect_url    = $field( 'redirect_url', '' );

// Decode fields JSON → PHP array.
$fields_json_raw = $field( 'fields', '[]' );
$fields_array    = json_decode( $fields_json_raw ?: '[]', true );
if ( ! is_array( $fields_array ) ) {
	$fields_array = [];
}

// Ensure sensible defaults for each field.
foreach ( $fields_array as &$f ) {
	$f = array_merge( [ 'name' => '', 'label' => '', 'type' => 'text', 'required' => false, 'placeholder' => '' ], $f );
}
unset( $f );

// List and tag IDs already assigned to this form.
$form_list_ids_str = $field( 'list_ids', '' );
$form_tag_ids_str  = $field( 'tag_ids', '' );
$form_list_ids     = array_filter( explode( ',', $form_list_ids_str ) );
$form_tag_ids      = array_filter( explode( ',', $form_tag_ids_str ) );

// Page title.
$page_title = $is_new
	? __( 'Crear nuevo formulario', 'myrock-mail-engine' )
	: __( 'Editar formulario', 'myrock-mail-engine' );

// Shortcode reference.
$shortcode = $form_id ? sprintf( '[mrme_form id="%d"]', $form_id ) : '';

// Available field types.
$field_types = [
	'text'     => __( 'Texto corto', 'myrock-mail-engine' ),
	'email'    => __( 'Correo electrónico', 'myrock-mail-engine' ),
	'phone'    => __( 'Teléfono', 'myrock-mail-engine' ),
	'textarea' => __( 'Texto largo', 'myrock-mail-engine' ),
	'select'   => __( 'Lista desplegable', 'myrock-mail-engine' ),
	'checkbox' => __( 'Casilla de verificación', 'myrock-mail-engine' ),
];
?>
<div class="wrap mrme-wrap">

	<h1 class="mrme-page-title">
		<span class="mrme-logo-mark">&#9679;</span>
		<?php echo esc_html( $page_title ); ?>
		<?php if ( $shortcode ) : ?>
		<span class="mrme-shortcode-badge" title="<?php esc_attr_e( 'Copia este shortcode para insertar el formulario en cualquier página', 'myrock-mail-engine' ); ?>">
			<?php echo esc_html( $shortcode ); ?>
			<button type="button" class="mrme-copy-shortcode" data-code="<?php echo esc_attr( $shortcode ); ?>" title="<?php esc_attr_e( 'Copiar', 'myrock-mail-engine' ); ?>">
				<span class="dashicons dashicons-clipboard"></span>
			</button>
		</span>
		<?php endif; ?>
	</h1>

	<?php if ( $notice ) : ?>
	<div class="notice notice-<?php echo esc_attr( $notice['type'] ); ?> is-dismissible">
		<p><?php echo esc_html( $notice['message'] ); ?></p>
	</div>
	<?php endif; ?>

	<form method="POST" action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>" id="mrme-form-edit" novalidate>
		<?php wp_nonce_field( 'mrme_save_form' ); ?>
		<input type="hidden" name="action" value="mrme_save_form">
		<input type="hidden" name="form_id" value="<?php echo esc_attr( $form_id ); ?>">
		<!-- Fields JSON is written here by JS before submit -->
		<input type="hidden" name="fields" id="mrme-fields-json" value="<?php echo esc_attr( $fields_json_raw ); ?>">

		<div class="mrme-form-builder-layout">

			<!-- ============================================================ -->
			<!-- Left column: builder                                         -->
			<!-- ============================================================ -->
			<div class="mrme-form-builder-main">

				<!-- Form name + status -->
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title">
						<span class="dashicons dashicons-feedback"></span>
						<?php esc_html_e( 'Datos del formulario', 'myrock-mail-engine' ); ?>
					</h2>
					<div class="mrme-form-section__row mrme-form-section__row--cols">
						<div style="flex:2">
							<label class="mrme-form-section__label" for="mrme-form-name">
								<?php esc_html_e( 'Nombre interno del formulario', 'myrock-mail-engine' ); ?> <span>*</span>
							</label>
							<input
								type="text"
								id="mrme-form-name"
								name="name"
								class="mrme-form-section__input large-text"
								value="<?php echo esc_attr( $form_name ); ?>"
								required
								placeholder="<?php esc_attr_e( 'Ej: Formulario de suscripción del blog', 'myrock-mail-engine' ); ?>"
							>
							<p class="description"><?php esc_html_e( 'Solo visible en el panel de administración.', 'myrock-mail-engine' ); ?></p>
						</div>
						<div>
							<label class="mrme-form-section__label" for="mrme-form-status">
								<?php esc_html_e( 'Estado', 'myrock-mail-engine' ); ?>
							</label>
							<select id="mrme-form-status" name="status" class="mrme-form-section__input">
								<option value="active"   <?php selected( $form_status, 'active' ); ?>><?php esc_html_e( 'Activo', 'myrock-mail-engine' ); ?></option>
								<option value="inactive" <?php selected( $form_status, 'inactive' ); ?>><?php esc_html_e( 'Inactivo', 'myrock-mail-engine' ); ?></option>
							</select>
						</div>
					</div>
				</div>

				<!-- Visual field builder -->
				<div class="mrme-form-section mrme-form-section--card" id="mrme-field-builder">
					<div class="mrme-form-section__header-row">
						<h2 class="mrme-form-section__title">
							<span class="dashicons dashicons-list-view"></span>
							<?php esc_html_e( 'Campos del formulario', 'myrock-mail-engine' ); ?>
						</h2>
						<button type="button" id="mrme-add-field-btn" class="button button-primary">
							<span class="dashicons dashicons-plus-alt2"></span>
							<?php esc_html_e( 'Agregar campo', 'myrock-mail-engine' ); ?>
						</button>
					</div>

					<p class="description" style="margin-bottom:12px;">
						<?php esc_html_e( 'Arrastra para reordenar. El campo "Correo electrónico" es obligatorio y no se puede eliminar.', 'myrock-mail-engine' ); ?>
					</p>

					<div id="mrme-fields-list" class="mrme-fields-list">
						<!-- Fields rendered by JS from initial JSON -->
					</div>

					<!-- Add field panel (hidden by default) -->
					<div id="mrme-add-field-panel" class="mrme-add-field-panel" style="display:none;">
						<h3><?php esc_html_e( 'Nuevo campo', 'myrock-mail-engine' ); ?></h3>
						<div class="mrme-add-field-panel__grid">
							<div>
								<label><?php esc_html_e( 'Tipo de campo', 'myrock-mail-engine' ); ?></label>
								<select id="mrme-new-field-type">
									<?php foreach ( $field_types as $val => $label ) : ?>
									<option value="<?php echo esc_attr( $val ); ?>"><?php echo esc_html( $label ); ?></option>
									<?php endforeach; ?>
								</select>
							</div>
							<div>
								<label><?php esc_html_e( 'Etiqueta visible', 'myrock-mail-engine' ); ?></label>
								<input type="text" id="mrme-new-field-label" placeholder="<?php esc_attr_e( 'Ej: Tu nombre', 'myrock-mail-engine' ); ?>">
							</div>
							<div>
								<label><?php esc_html_e( 'Nombre técnico (sin espacios)', 'myrock-mail-engine' ); ?></label>
								<input type="text" id="mrme-new-field-name" placeholder="<?php esc_attr_e( 'Ej: nombre', 'myrock-mail-engine' ); ?>">
							</div>
							<div>
								<label><?php esc_html_e( 'Placeholder', 'myrock-mail-engine' ); ?></label>
								<input type="text" id="mrme-new-field-placeholder" placeholder="<?php esc_attr_e( 'Ej: Escribe tu nombre…', 'myrock-mail-engine' ); ?>">
							</div>
						</div>
						<div class="mrme-add-field-panel__required">
							<label>
								<input type="checkbox" id="mrme-new-field-required">
								<?php esc_html_e( 'Campo obligatorio', 'myrock-mail-engine' ); ?>
							</label>
						</div>
						<div class="mrme-add-field-panel__actions">
							<button type="button" id="mrme-confirm-add-field" class="button button-primary">
								<?php esc_html_e( 'Agregar al formulario', 'myrock-mail-engine' ); ?>
							</button>
							<button type="button" id="mrme-cancel-add-field" class="button">
								<?php esc_html_e( 'Cancelar', 'myrock-mail-engine' ); ?>
							</button>
						</div>
					</div>
				</div>

				<!-- After-submission settings -->
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title">
						<span class="dashicons dashicons-yes-alt"></span>
						<?php esc_html_e( 'Después del envío', 'myrock-mail-engine' ); ?>
					</h2>

					<div class="mrme-form-section__row">
						<label class="mrme-form-section__label" for="mrme-success-msg">
							<?php esc_html_e( 'Mensaje de éxito', 'myrock-mail-engine' ); ?>
						</label>
						<textarea
							id="mrme-success-msg"
							name="success_message"
							class="mrme-form-section__textarea large-text"
							rows="3"
							placeholder="<?php esc_attr_e( '¡Gracias! Te has suscrito correctamente.', 'myrock-mail-engine' ); ?>"
						><?php echo esc_textarea( $success_message ); ?></textarea>
						<p class="description"><?php esc_html_e( 'Texto que verá el usuario tras suscribirse exitosamente.', 'myrock-mail-engine' ); ?></p>
					</div>

					<div class="mrme-form-section__row">
						<label class="mrme-form-section__label" for="mrme-redirect-url">
							<?php esc_html_e( 'Redirigir a URL (opcional)', 'myrock-mail-engine' ); ?>
						</label>
						<input
							type="url"
							id="mrme-redirect-url"
							name="redirect_url"
							class="mrme-form-section__input large-text"
							value="<?php echo esc_attr( $redirect_url ); ?>"
							placeholder="https://tusitio.com/gracias"
						>
						<p class="description"><?php esc_html_e( 'Si se especifica, el usuario será redirigido a esta URL en lugar de ver el mensaje de éxito.', 'myrock-mail-engine' ); ?></p>
					</div>

					<div class="mrme-form-section__row">
						<label>
							<input type="checkbox" name="double_optin" value="1" <?php checked( $double_optin, 1 ); ?>>
							<strong><?php esc_html_e( 'Activar doble opt-in', 'myrock-mail-engine' ); ?></strong>
						</label>
						<p class="description" style="margin-top:4px;">
							<?php esc_html_e( 'Con esta opción activada, el nuevo suscriptor recibirá un correo de confirmación y su suscripción no será efectiva hasta que haga clic en el enlace de confirmación.', 'myrock-mail-engine' ); ?>
						</p>
					</div>
				</div>

			</div><!-- /.mrme-form-builder-main -->

			<!-- ============================================================ -->
			<!-- Right column: sidebar                                        -->
			<!-- ============================================================ -->
			<div class="mrme-form-builder-sidebar">

				<!-- Save box -->
				<div class="mrme-form-section mrme-form-section--card mrme-submit-box">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Guardar', 'myrock-mail-engine' ); ?></h2>
					<button type="submit" class="button button-primary button-large" style="width:100%">
						<?php echo $is_new
							? esc_html__( 'Crear formulario', 'myrock-mail-engine' )
							: esc_html__( 'Guardar cambios', 'myrock-mail-engine' );
						?>
					</button>
					<div style="margin-top:8px;">
						<a href="<?php echo esc_url( $base_url ); ?>">&larr; <?php esc_html_e( 'Volver a formularios', 'myrock-mail-engine' ); ?></a>
					</div>
				</div>

				<!-- Shortcode info -->
				<?php if ( $shortcode ) : ?>
				<div class="mrme-form-section mrme-form-section--card mrme-shortcode-box">
					<h2 class="mrme-form-section__title">
						<span class="dashicons dashicons-editor-code"></span>
						<?php esc_html_e( 'Shortcode de inserción', 'myrock-mail-engine' ); ?>
					</h2>
					<p class="description">
						<?php esc_html_e( 'Pega este código en cualquier página, entrada o widget de texto para mostrar el formulario:', 'myrock-mail-engine' ); ?>
					</p>
					<div class="mrme-shortcode-display">
						<code id="mrme-shortcode-text"><?php echo esc_html( $shortcode ); ?></code>
						<button type="button" class="button mrme-copy-shortcode" data-code="<?php echo esc_attr( $shortcode ); ?>">
							<span class="dashicons dashicons-clipboard"></span>
							<?php esc_html_e( 'Copiar', 'myrock-mail-engine' ); ?>
						</button>
					</div>
				</div>
				<?php endif; ?>

				<!-- Lists -->
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title">
						<span class="dashicons dashicons-groups"></span>
						<?php esc_html_e( 'Suscribir a listas', 'myrock-mail-engine' ); ?>
					</h2>
					<p class="description"><?php esc_html_e( 'Los nuevos suscriptores serán añadidos a las listas seleccionadas.', 'myrock-mail-engine' ); ?></p>
					<?php if ( ! empty( $lists ) ) : ?>
					<div class="mrme-checkbox-grid" style="margin-top:10px;">
						<?php foreach ( $lists as $lst ) :
							$lst_id      = is_array( $lst ) ? (int) $lst['id']   : (int) $lst->id;
							$lst_name    = is_array( $lst ) ? $lst['name'] : $lst->name;
							$is_selected = in_array( (string) $lst_id, array_map( 'strval', $form_list_ids ), true );
						?>
						<label class="mrme-checkbox-grid__item">
							<input type="checkbox" name="list_ids[]" value="<?php echo esc_attr( $lst_id ); ?>" <?php checked( $is_selected ); ?>>
							<?php echo esc_html( $lst_name ); ?>
						</label>
						<?php endforeach; ?>
					</div>
					<?php else : ?>
					<p class="mrme-muted" style="margin-top:8px;">
						<?php esc_html_e( 'No hay listas disponibles.', 'myrock-mail-engine' ); ?>
						<a href="<?php echo esc_url( admin_url( 'admin.php?page=mrme-lists&action=new' ) ); ?>"><?php esc_html_e( 'Crear lista', 'myrock-mail-engine' ); ?></a>
					</p>
					<?php endif; ?>
				</div>

				<!-- Tags -->
				<?php if ( ! empty( $tags ) ) : ?>
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title">
						<span class="dashicons dashicons-tag"></span>
						<?php esc_html_e( 'Asignar etiquetas', 'myrock-mail-engine' ); ?>
					</h2>
					<p class="description"><?php esc_html_e( 'Los nuevos suscriptores recibirán automáticamente estas etiquetas.', 'myrock-mail-engine' ); ?></p>
					<div class="mrme-checkbox-grid" style="margin-top:10px;">
						<?php foreach ( $tags as $tag ) :
							$tag_id      = is_array( $tag ) ? (int) $tag['id']   : (int) $tag->id;
							$tag_name    = is_array( $tag ) ? $tag['name'] : $tag->name;
							$is_selected = in_array( (string) $tag_id, array_map( 'strval', $form_tag_ids ), true );
						?>
						<label class="mrme-checkbox-grid__item">
							<input type="checkbox" name="tag_ids[]" value="<?php echo esc_attr( $tag_id ); ?>" <?php checked( $is_selected ); ?>>
							<?php echo esc_html( $tag_name ); ?>
						</label>
						<?php endforeach; ?>
					</div>
				</div>
				<?php endif; ?>

				<!-- Live preview -->
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title">
						<span class="dashicons dashicons-visibility"></span>
						<?php esc_html_e( 'Vista previa', 'myrock-mail-engine' ); ?>
					</h2>
					<p class="description"><?php esc_html_e( 'Así verán el formulario tus visitantes.', 'myrock-mail-engine' ); ?></p>
					<div id="mrme-form-preview" class="mrme-form-preview">
						<!-- Rendered by JS -->
					</div>
				</div>

			</div><!-- /.mrme-form-builder-sidebar -->

		</div><!-- /.mrme-form-builder-layout -->
	</form>

</div><!-- /.wrap.mrme-wrap -->

<script type="text/javascript">
(function($) {
	'use strict';

	// --- State ---
	var fields = <?php echo wp_json_encode( array_values( $fields_array ) ); ?>;

	// Field type labels
	var typeLabels = {
		text:     '<?php esc_js( _e( 'Texto corto', 'myrock-mail-engine' ) ); ?>',
		email:    '<?php esc_js( _e( 'Correo electrónico', 'myrock-mail-engine' ) ); ?>',
		phone:    '<?php esc_js( _e( 'Teléfono', 'myrock-mail-engine' ) ); ?>',
		textarea: '<?php esc_js( _e( 'Texto largo', 'myrock-mail-engine' ) ); ?>',
		select:   '<?php esc_js( _e( 'Lista desplegable', 'myrock-mail-engine' ) ); ?>',
		checkbox: '<?php esc_js( _e( 'Casilla', 'myrock-mail-engine' ) ); ?>'
	};

	function typeLabel(type) {
		return typeLabels[type] || type;
	}

	// --- Render field list ---
	function renderFields() {
		var $list = $('#mrme-fields-list').empty();
		if (!fields.length) {
			$list.append('<p class="mrme-muted" style="padding:16px;"><?php esc_js( _e( 'No hay campos todavía. Haz clic en «Agregar campo» para comenzar.', 'myrock-mail-engine' ) ); ?></p>');
			renderPreview();
			return;
		}
		$.each(fields, function(idx, f) {
			var isEmail = (f.name === 'email' || f.type === 'email');
			var $row = $(
				'<div class="mrme-field-row" data-idx="' + idx + '">' +
					'<span class="mrme-field-row__handle dashicons dashicons-move" title="Arrastrar para ordenar"></span>' +
					'<div class="mrme-field-row__info">' +
						'<span class="mrme-field-row__label">' + escHtml(f.label || f.name) + '</span>' +
						'<span class="mrme-field-row__meta mrme-muted">' + typeLabel(f.type) + (f.required ? ' · <?php esc_js( _e( 'Obligatorio', 'myrock-mail-engine' ) ); ?>' : '') + '</span>' +
					'</div>' +
					'<div class="mrme-field-row__actions">' +
						(!isEmail ? '<button type="button" class="mrme-field-edit button button-small"><?php esc_js( _e( 'Editar', 'myrock-mail-engine' ) ); ?></button>' : '') +
						(!isEmail ? '<button type="button" class="mrme-field-delete button button-small mrme-btn--danger"><?php esc_js( _e( 'Eliminar', 'myrock-mail-engine' ) ); ?></button>' : '<span class="mrme-badge mrme-badge--subscribed"><?php esc_js( _e( 'Requerido', 'myrock-mail-engine' ) ); ?></span>') +
					'</div>' +
				'</div>'
			);
			$list.append($row);
		});

		// Sortable via jQuery UI (loaded in WP admin).
		if ($.fn.sortable) {
			$list.sortable({
				handle: '.mrme-field-row__handle',
				axis: 'y',
				update: function() {
					var newOrder = [];
					$list.find('.mrme-field-row').each(function() {
						newOrder.push(fields[parseInt($(this).data('idx'))]);
					});
					fields = newOrder;
					renderFields();
				}
			});
		}

		syncJson();
		renderPreview();
	}

	// --- Sync hidden JSON field ---
	function syncJson() {
		$('#mrme-fields-json').val(JSON.stringify(fields));
	}

	// --- Live preview ---
	function renderPreview() {
		var $p = $('#mrme-form-preview').empty();
		if (!fields.length) {
			$p.html('<p class="mrme-muted"><?php esc_js( _e( 'Agrega campos para ver la vista previa.', 'myrock-mail-engine' ) ); ?></p>');
			return;
		}
		var html = '<div class="mrme-preview-form">';
		$.each(fields, function(i, f) {
			html += '<div class="mrme-preview-field">';
			html += '<label class="mrme-preview-field__label">' + escHtml(f.label || f.name);
			if (f.required) html += ' <span style="color:#c33;">*</span>';
			html += '</label>';
			if (f.type === 'textarea') {
				html += '<textarea class="mrme-preview-field__input" disabled placeholder="' + escAttr(f.placeholder || '') + '" rows="3"></textarea>';
			} else if (f.type === 'checkbox') {
				html += '<label><input type="checkbox" disabled> ' + escHtml(f.label || f.name) + '</label>';
			} else if (f.type === 'select') {
				html += '<select class="mrme-preview-field__input" disabled><option><?php esc_js( _e( 'Seleccionar…', 'myrock-mail-engine' ) ); ?></option></select>';
			} else {
				html += '<input type="' + escAttr(f.type) + '" class="mrme-preview-field__input" disabled placeholder="' + escAttr(f.placeholder || '') + '">';
			}
			html += '</div>';
		});
		html += '<button type="button" class="mrme-preview-submit button button-primary" disabled><?php esc_js( _e( 'Suscribirme', 'myrock-mail-engine' ) ); ?></button>';
		html += '</div>';
		$p.html(html);
	}

	// --- Escape helpers ---
	function escHtml(str) {
		return $('<div>').text(str).html();
	}
	function escAttr(str) {
		return $('<div>').text(str).html().replace(/"/g, '&quot;');
	}

	// --- Events: open/close add-field panel ---
	$('#mrme-add-field-btn').on('click', function() {
		$('#mrme-add-field-panel').slideDown(150);
		$('#mrme-new-field-label').focus();
	});
	$('#mrme-cancel-add-field').on('click', function() {
		$('#mrme-add-field-panel').slideUp(150);
		clearNewFieldForm();
	});

	// Auto-generate field name from label
	$('#mrme-new-field-label').on('input', function() {
		var name = $(this).val()
			.toLowerCase()
			.normalize('NFD').replace(/[\u0300-\u036f]/g, '')
			.replace(/[^a-z0-9]+/g, '_')
			.replace(/^_|_$/g, '');
		$('#mrme-new-field-name').val(name);
	});

	// Confirm add field
	$('#mrme-confirm-add-field').on('click', function() {
		var type        = $('#mrme-new-field-type').val();
		var label       = $('#mrme-new-field-label').val().trim();
		var name        = $('#mrme-new-field-name').val().trim();
		var placeholder = $('#mrme-new-field-placeholder').val().trim();
		var required    = $('#mrme-new-field-required').is(':checked');

		if (!label) { alert('<?php esc_js( _e( 'La etiqueta del campo es obligatoria.', 'myrock-mail-engine' ) ); ?>'); return; }
		if (!name)  { alert('<?php esc_js( _e( 'El nombre técnico del campo es obligatorio.', 'myrock-mail-engine' ) ); ?>'); return; }

		// Prevent duplicate names
		var exists = fields.some(function(f) { return f.name === name; });
		if (exists) { alert('<?php esc_js( _e( 'Ya existe un campo con ese nombre. Elige otro.', 'myrock-mail-engine' ) ); ?>'); return; }

		fields.push({ name: name, label: label, type: type, required: required, placeholder: placeholder });
		$('#mrme-add-field-panel').slideUp(150);
		clearNewFieldForm();
		renderFields();
	});

	function clearNewFieldForm() {
		$('#mrme-new-field-type').val('text');
		$('#mrme-new-field-label, #mrme-new-field-name, #mrme-new-field-placeholder').val('');
		$('#mrme-new-field-required').prop('checked', false);
	}

	// Delete field
	$('#mrme-fields-list').on('click', '.mrme-field-delete', function() {
		var idx = parseInt($(this).closest('.mrme-field-row').data('idx'));
		if (!confirm('<?php esc_js( _e( '¿Eliminar este campo?', 'myrock-mail-engine' ) ); ?>')) return;
		fields.splice(idx, 1);
		renderFields();
	});

	// Edit field (inline toggle)
	$('#mrme-fields-list').on('click', '.mrme-field-edit', function() {
		var $row = $(this).closest('.mrme-field-row');
		var idx  = parseInt($row.data('idx'));
		var f    = fields[idx];

		// Remove existing inline editor if open
		$('.mrme-field-inline-editor').remove();
		if ($row.find('.mrme-field-inline-editor').length) return;

		var typeOptions = '';
		$.each(typeLabels, function(val, lbl) {
			typeOptions += '<option value="' + val + '"' + (f.type === val ? ' selected' : '') + '>' + escHtml(lbl) + '</option>';
		});

		var $editor = $(
			'<div class="mrme-field-inline-editor">' +
				'<div class="mrme-inline-grid">' +
					'<div><label><?php esc_js( _e( 'Tipo', 'myrock-mail-engine' ) ); ?></label><select class="ie-type">' + typeOptions + '</select></div>' +
					'<div><label><?php esc_js( _e( 'Etiqueta', 'myrock-mail-engine' ) ); ?></label><input type="text" class="ie-label" value="' + escAttr(f.label) + '"></div>' +
					'<div><label><?php esc_js( _e( 'Nombre', 'myrock-mail-engine' ) ); ?></label><input type="text" class="ie-name" value="' + escAttr(f.name) + '"></div>' +
					'<div><label><?php esc_js( _e( 'Placeholder', 'myrock-mail-engine' ) ); ?></label><input type="text" class="ie-placeholder" value="' + escAttr(f.placeholder || '') + '"></div>' +
				'</div>' +
				'<label style="margin-top:8px;display:block;"><input type="checkbox" class="ie-required"' + (f.required ? ' checked' : '') + '> <?php esc_js( _e( 'Obligatorio', 'myrock-mail-engine' ) ); ?></label>' +
				'<div class="mrme-inline-editor-actions" style="margin-top:10px;">' +
					'<button type="button" class="ie-save button button-primary button-small"><?php esc_js( _e( 'Guardar', 'myrock-mail-engine' ) ); ?></button> ' +
					'<button type="button" class="ie-cancel button button-small"><?php esc_js( _e( 'Cancelar', 'myrock-mail-engine' ) ); ?></button>' +
				'</div>' +
			'</div>'
		);

		$row.after($editor);

		$editor.find('.ie-cancel').on('click', function() { $editor.remove(); });
		$editor.find('.ie-save').on('click', function() {
			fields[idx].type        = $editor.find('.ie-type').val();
			fields[idx].label       = $editor.find('.ie-label').val().trim();
			fields[idx].name        = $editor.find('.ie-name').val().trim();
			fields[idx].placeholder = $editor.find('.ie-placeholder').val().trim();
			fields[idx].required    = $editor.find('.ie-required').is(':checked');
			$editor.remove();
			renderFields();
		});
	});

	// Sync JSON before submit
	$('#mrme-form-edit').on('submit', function() {
		syncJson();
	});

	// Copy shortcode button
	$(document).on('click', '.mrme-copy-shortcode', function() {
		var code = $(this).data('code');
		navigator.clipboard.writeText(code).then(function() {
			// brief visual feedback
		});
	});

	// Copy merge tag button
	$(document).on('click', '.mrme-tag-copy', function() {
		var tag = $(this).data('tag');
		if (navigator.clipboard) {
			navigator.clipboard.writeText(tag);
		}
	});

	// Init
	renderFields();

})(jQuery);
</script>
