<?php
/**
 * Site footer template part.
 *
 * @package MyRock_Theme
 */
defined( 'ABSPATH' ) || exit;
?>
<footer class="mr-site-footer" role="contentinfo">
  <div class="mr-container mr-footer-inner">

    <div class="mr-footer-brand">
      <a href="<?php echo esc_url( home_url( '/' ) ); ?>" class="mr-site-logo" aria-label="MyRock">
        <div class="mr-logo-mark" aria-hidden="true">MR</div>
        <span class="mr-logo-name">MyRock</span>
      </a>
      <p class="mr-footer-tagline">
        Firma digital de alto nivel en Culiacán, Sinaloa. Diseño, web y sistemas para marcas que no se conforman con lo ordinario.
      </p>
    </div>

    <div class="mr-footer-links">
      <div class="mr-footer-col">
        <div class="mr-footer-col-title">Blog</div>
        <ul>
          <li><a href="<?php echo esc_url( home_url( '/blog/' ) ); ?>">Todos los artículos</a></li>
          <?php
          $cats = get_categories( [ 'hide_empty' => true, 'number' => 4 ] );
          foreach ( $cats as $cat ) :
          ?>
            <li><a href="<?php echo esc_url( get_category_link( $cat->term_id ) ); ?>"><?php echo esc_html( $cat->name ); ?></a></li>
          <?php endforeach; ?>
        </ul>
      </div>
      <div class="mr-footer-col">
        <div class="mr-footer-col-title">Empresa</div>
        <ul>
          <li><a href="<?php echo esc_url( home_url( '/' ) ); ?>">Inicio</a></li>
          <li><a href="<?php echo esc_url( home_url( '/#services' ) ); ?>">Servicios</a></li>
          <li><a href="<?php echo esc_url( home_url( '/#cases' ) ); ?>">Portafolio</a></li>
          <li><a href="<?php echo esc_url( home_url( '/#contact' ) ); ?>">Contacto</a></li>
        </ul>
      </div>
      <div class="mr-footer-col">
        <div class="mr-footer-col-title">Contacto</div>
        <ul>
          <li><a href="mailto:hola@myrock.com.mx">hola@myrock.com.mx</a></li>
          <li><span>Culiacán, Sinaloa, México</span></li>
          <li><span>Operación remota · Todo México</span></li>
        </ul>
      </div>
    </div>

  </div>

  <div class="mr-footer-bottom">
    <div class="mr-container mr-footer-bottom-inner">
      <span>© <?php echo date( 'Y' ); ?> MyRock. Hecho con precisión en Culiacán, Sinaloa.</span>
      <nav aria-label="Legal">
        <a href="#">Privacidad</a>
        <a href="#">Términos</a>
      </nav>
    </div>
  </div>
</footer>
