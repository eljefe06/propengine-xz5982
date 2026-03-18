<?php
defined( 'ABSPATH' ) || exit;

$webhook_url = get_rest_url( null, 'mrls/v1/webhook' );
?>
<div class="wrap">
	<h1><?php esc_html_e( 'Configuración — Licencias MyRock', 'mrls' ); ?></h1>

	<?php if ( isset( $_GET['updated'] ) ) : ?>
		<div class="notice notice-success is-dismissible"><p>✅ Configuración guardada.</p></div>
	<?php endif; ?>

	<form method="post" action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>">
		<?php wp_nonce_field( 'mrls_save_settings' ); ?>
		<input type="hidden" name="action" value="mrls_save_settings">

		<!-- ── MercadoPago ────────────────────────────────────────── -->
		<h2>MercadoPago</h2>
		<table class="form-table" role="presentation">
			<tr>
				<th scope="row"><label for="mrls_mp_access_token">Access Token de producción</label></th>
				<td>
					<input type="password" id="mrls_mp_access_token" name="mrls_mp_access_token"
						   value="<?php echo esc_attr( get_option( 'mrls_mp_access_token', '' ) ); ?>"
						   class="regular-text" autocomplete="off">
					<p class="description">
						Obtenerlo en <a href="https://www.mercadopago.com.mx/settings/account/credentials" target="_blank">
						MercadoPago → Credenciales → Producción → Access Token</a>.
					</p>
				</td>
			</tr>
			<tr>
				<th scope="row">URL del Webhook</th>
				<td>
					<code><?php echo esc_html( $webhook_url ); ?></code>
					<button type="button" class="button button-small"
							onclick="navigator.clipboard.writeText('<?php echo esc_js( $webhook_url ); ?>').then(()=>this.textContent='✅ Copiado').catch(()=>{})">
						Copiar
					</button>
					<p class="description">
						Configura esta URL en
						<a href="https://www.mercadopago.com.mx/developers/panel/webhooks" target="_blank">
							MercadoPago → Tu negocio → Webhooks → Agregar URL
						</a>.<br>
						Suscríbete a los eventos: <code>subscription_preapproval</code> y <code>payment</code>.
					</p>
				</td>
			</tr>
		</table>

		<!-- ── Plan Mensual ──────────────────────────────────────── -->
		<h2>Plan Mensual</h2>
		<table class="form-table" role="presentation">
			<tr>
				<th scope="row"><label for="mrls_price_monthly">Precio</label></th>
				<td>
					<input type="number" id="mrls_price_monthly" name="mrls_price_monthly"
						   value="<?php echo esc_attr( get_option( 'mrls_price_monthly', 299 ) ); ?>"
						   step="0.01" class="small-text"> MXN / mes
				</td>
			</tr>
			<tr>
				<th scope="row"><label for="mrls_mp_plan_monthly_id">ID del plan en MP</label></th>
				<td>
					<input type="text" id="mrls_mp_plan_monthly_id" name="mrls_mp_plan_monthly_id"
						   value="<?php echo esc_attr( get_option( 'mrls_mp_plan_monthly_id', '' ) ); ?>"
						   class="regular-text" placeholder="2c93808493...">
					<p class="description">El <code>id</code> del <code>preapproval_plan</code> que creaste en MP.</p>
				</td>
			</tr>
			<tr>
				<th scope="row"><label for="mrls_mp_checkout_monthly">URL de checkout</label></th>
				<td>
					<input type="url" id="mrls_mp_checkout_monthly" name="mrls_mp_checkout_monthly"
						   value="<?php echo esc_attr( get_option( 'mrls_mp_checkout_monthly', '' ) ); ?>"
						   class="large-text" placeholder="https://www.mercadopago.com.mx/subscriptions/checkout?preapproval_plan_id=...">
					<p class="description">La URL de checkout que MP genera para el plan mensual. Los clientes son redirigidos aquí al hacer clic en «Suscribirme mensual».</p>
				</td>
			</tr>
		</table>

		<!-- ── Plan Anual ────────────────────────────────────────── -->
		<h2>Plan Anual</h2>
		<table class="form-table" role="presentation">
			<tr>
				<th scope="row"><label for="mrls_price_annual">Precio</label></th>
				<td>
					<input type="number" id="mrls_price_annual" name="mrls_price_annual"
						   value="<?php echo esc_attr( get_option( 'mrls_price_annual', 2499 ) ); ?>"
						   step="0.01" class="small-text"> MXN / año
				</td>
			</tr>
			<tr>
				<th scope="row"><label for="mrls_mp_plan_annual_id">ID del plan en MP</label></th>
				<td>
					<input type="text" id="mrls_mp_plan_annual_id" name="mrls_mp_plan_annual_id"
						   value="<?php echo esc_attr( get_option( 'mrls_mp_plan_annual_id', '' ) ); ?>"
						   class="regular-text">
				</td>
			</tr>
			<tr>
				<th scope="row"><label for="mrls_mp_checkout_annual">URL de checkout</label></th>
				<td>
					<input type="url" id="mrls_mp_checkout_annual" name="mrls_mp_checkout_annual"
						   value="<?php echo esc_attr( get_option( 'mrls_mp_checkout_annual', '' ) ); ?>"
						   class="large-text">
				</td>
			</tr>
		</table>

		<?php submit_button( 'Guardar configuración' ); ?>
	</form>

	<!-- ── Instrucciones ─────────────────────────────────────────── -->
	<hr>
	<h2>📋 Pasos para configurar MercadoPago</h2>
	<ol style="max-width:640px;line-height:1.8">
		<li>Ve a <a href="https://www.mercadopago.com.mx/developers/panel/app" target="_blank">MP Developers Panel</a> y crea una aplicación.</li>
		<li>En <strong>Credenciales → Producción</strong>, copia el <strong>Access Token</strong> y pégalo arriba.</li>
		<li>Crea los planes de suscripción vía API o desde el panel:<br>
			<code>POST https://api.mercadopago.com/preapproval_plan</code><br>
			con <code>frequency: 1</code>, <code>frequency_type: "months"</code> (mensual) o <code>"years"</code> (anual), <code>transaction_amount</code> y <code>currency_id: "MXN"</code>.
		</li>
		<li>Copia el <code>id</code> del plan y su <code>init_point</code> (URL de checkout) en los campos de arriba.</li>
		<li>En <strong>Tu negocio → Webhooks → Agregar URL</strong>, pega la URL del webhook de arriba y selecciona los eventos <code>subscription_preapproval</code> y <code>payment</code>.</li>
		<li>¡Listo! Cada pago aprobado genera o renueva la licencia automáticamente y la envía por correo.</li>
	</ol>
</div>
