<?php
defined( 'ABSPATH' ) || exit;

use MyRock\LicenseServer\Models\License;

$licenses = License::find_all( 50, 1 );
$total    = License::count();
?>
<div class="wrap">
	<h1><?php esc_html_e( 'Licencias — MyRock Mail Engine Pro', 'mrls' ); ?></h1>

	<?php if ( isset( $_GET['created'] ) ) : ?>
		<div class="notice notice-success is-dismissible"><p>✅ Licencia creada y enviada por correo.</p></div>
	<?php endif; ?>
	<?php if ( isset( $_GET['cancelled'] ) ) : ?>
		<div class="notice notice-warning is-dismissible"><p>Licencia cancelada.</p></div>
	<?php endif; ?>

	<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;margin-top:16px">

		<!-- Crear licencia manual -->
		<div style="background:#fff;padding:20px;border:1px solid #ddd;border-radius:6px;min-width:300px">
			<h3 style="margin-top:0">➕ Crear licencia manual</h3>
			<form method="post" action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>">
				<?php wp_nonce_field( 'mrls_create_license' ); ?>
				<input type="hidden" name="action" value="mrls_create_license">
				<p>
					<label style="font-weight:600">Email del cliente<br>
						<input type="email" name="email" required style="width:100%;margin-top:4px">
					</label>
				</p>
				<p>
					<label style="font-weight:600">Plan<br>
						<select name="plan" style="width:100%;margin-top:4px">
							<option value="monthly">Mensual — $299 MXN/mes</option>
							<option value="annual">Anual — $2,499 MXN/año</option>
						</select>
					</label>
				</p>
				<p>
					<label style="font-weight:600">Notas internas<br>
						<textarea name="notes" style="width:100%;margin-top:4px" rows="2"></textarea>
					</label>
				</p>
				<button type="submit" class="button button-primary">Crear y enviar por correo</button>
			</form>
		</div>

		<!-- Resumen -->
		<div style="background:#f0f6ff;padding:20px;border:1px solid #c5d8f5;border-radius:6px;min-width:180px">
			<p style="margin:0;font-size:13px;color:#555">Total de licencias</p>
			<p style="margin:4px 0 0;font-size:32px;font-weight:700;color:#1B2980"><?php echo esc_html( $total ); ?></p>
		</div>

	</div>

	<table class="widefat fixed striped" style="margin-top:24px">
		<thead>
			<tr>
				<th style="width:200px">Llave</th>
				<th>Email</th>
				<th style="width:80px">Plan</th>
				<th style="width:80px">Estado</th>
				<th style="width:100px">Expira</th>
				<th style="width:100px">Renovada</th>
				<th>Suscripción MP</th>
				<th style="width:80px">Acción</th>
			</tr>
		</thead>
		<tbody>
		<?php if ( empty( $licenses ) ) : ?>
			<tr><td colspan="8" style="text-align:center;color:#888;padding:30px">Sin licencias aún.</td></tr>
		<?php else : ?>
			<?php foreach ( $licenses as $lic ) : ?>
				<?php
				$status_colors = [
					'active'    => '#16a34a',
					'cancelled' => '#dc2626',
					'expired'   => '#d97706',
					'pending'   => '#6b7280',
				];
				$color = $status_colors[ $lic->status ] ?? '#6b7280';
				?>
				<tr>
					<td><code style="font-size:11px"><?php echo esc_html( $lic->license_key ); ?></code></td>
					<td><?php echo esc_html( $lic->email ); ?></td>
					<td><?php echo esc_html( $lic->plan ); ?></td>
					<td><span style="color:<?php echo esc_attr( $color ); ?>;font-weight:700"><?php echo esc_html( $lic->status ); ?></span></td>
					<td><?php echo $lic->expires_at ? esc_html( wp_date( 'd/m/Y', strtotime( $lic->expires_at ) ) ) : '—'; ?></td>
					<td><?php echo $lic->last_renewed_at ? esc_html( wp_date( 'd/m/Y', strtotime( $lic->last_renewed_at ) ) ) : '—'; ?></td>
					<td><small style="color:#888"><?php echo esc_html( $lic->mp_subscription_id ?? '—' ); ?></small></td>
					<td>
						<?php if ( 'active' === $lic->status ) : ?>
						<form method="post" action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>"
							  onsubmit="return confirm('¿Cancelar esta licencia?')">
							<?php wp_nonce_field( 'mrls_cancel_license' ); ?>
							<input type="hidden" name="action" value="mrls_cancel_license">
							<input type="hidden" name="license_id" value="<?php echo esc_attr( $lic->id ); ?>">
							<button type="submit" class="button button-small">Cancelar</button>
						</form>
						<?php endif; ?>
					</td>
				</tr>
			<?php endforeach; ?>
		<?php endif; ?>
		</tbody>
	</table>
</div>
