<?php
/**
 * Newsletter CTA strip shown below every archive and single post.
 *
 * @package MyRock_Theme
 */
defined( 'ABSPATH' ) || exit;
?>
<section class="mr-newsletter-strip">
  <div class="mr-container">
    <div class="mr-newsletter-strip__inner">
      <div class="mr-newsletter-strip__copy">
        <span class="mr-newsletter-strip__label">Newsletter</span>
        <h2 class="mr-newsletter-strip__title">Estrategia digital, directo a tu bandeja.</h2>
        <p class="mr-newsletter-strip__sub">Sin spam. Sin frecuencia ansiosa. Solo artículos cuando valga la pena.</p>
      </div>
      <div class="mr-newsletter-strip__form-wrap">
        <?php if ( function_exists( 'mrme_form_shortcode' ) ) : ?>
          <?php echo do_shortcode( '[mrme_form id="1" button="Suscribirme" dark="true"]' ); ?>
        <?php else : ?>
          <form class="mr-newsletter-form" action="#" method="post">
            <input type="email" name="email" placeholder="tu@email.com" required class="mr-newsletter-input">
            <button type="submit" class="mr-newsletter-btn">Suscribirme →</button>
          </form>
        <?php endif; ?>
      </div>
    </div>
  </div>
</section>
