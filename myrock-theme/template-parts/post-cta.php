<?php
/**
 * CTA section at the end of every post.
 *
 * @package MyRock_Theme
 */
defined( 'ABSPATH' ) || exit;
?>
<section class="mr-post-cta">
  <div class="mr-container mr-container--narrow">
    <div class="mr-post-cta__inner">
      <div class="mr-post-cta__text">
        <span class="mr-post-cta__label">¿Tienes un proyecto?</span>
        <h2 class="mr-post-cta__headline">
          Transforma tu presencia digital con <em>criterio y precisión</em>.
        </h2>
        <p class="mr-post-cta__sub">
          Trabajamos con marcas que tratan su presencia digital con el mismo nivel de exigencia que su producto.
        </p>
      </div>
      <div class="mr-post-cta__actions">
        <a href="<?php echo esc_url( home_url( '/#contact' ) ); ?>" class="mr-btn-primary">
          Agendar una llamada →
        </a>
        <?php if ( function_exists( 'mrme_form_shortcode' ) ) : ?>
          <p class="mr-post-cta__or">o suscríbete al newsletter</p>
          <div class="mr-post-cta__form">
            <?php echo do_shortcode( '[mrme_form id="1" button="Suscribirme"]' ); ?>
          </div>
        <?php endif; ?>
      </div>
    </div>
  </div>
</section>
