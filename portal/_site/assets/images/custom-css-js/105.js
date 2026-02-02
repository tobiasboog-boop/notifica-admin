<!-- start Simple Custom CSS and JS -->
<script type="text/javascript">
jQuery(document).ready(function( $ ){

  
  $('.container.main-timeline.post-timeline .tag span ').each(function(){
    var timeLine = $(this).text().slice(3);
    $(this).html(timeLine);
  })
  var i = 0;  
$('.timestamp').each(function(){
  i++
  $(this).html(i)
})
 
$('.bouw-link').on('click',function(){
  $('.navigation-selector .col').addClass('transition-out')
  $('.fading-nav').addClass('transition-out')
  setTimeout(function(){bouwNav();},600);
})  
function bouwNav () {
  window.location.href = "/bouw-nav/";
}
$('.bouw-synthess-link').on('click',function(){
    $('.navigation-selector .col').addClass('transition-out')
    setTimeout(function(){bouwNavSoft();},600);
})  
function bouwNavSoft () {
  window.location.href = "/bouw-nav/bouw-syntess/";
}

  
  
$('.navigation-selector-two .navigation-back-button-one').on('click', function(){
    history.back();
})
$('.navigation-selector-three .navigation-back-button-one').on('click', function(){
  console.log('backy')
	history.go(-2);
})  
$('.navigation-selector-three .navigation-back-button-two').on('click', function(){
	history.back();
})   
  
$('.product-small').each(function(){
  $(this).find('.price').append($(this).find(".add-to-cart-button"));
})
$('.product-section .subscription-details').html('p.m.'); 
$('.product-section .subscription-details').append('<p class="trail-period">Inc. 30 dagen proef periode. Deze start na de oplevering van de rapportages. Verwachte oplevertermijn 14 dagen.</p>')

$('.cart_item .subscription-details').html('p.m.');
$('.cart_item .subscription-details').append('<p class="trail-period">Inc. 30 dagen proef periode. Deze start na de oplevering van de rapportages. Verwachte oplevertermijn 14 dagen.</p>')

//$('html .woocommerce-checkout-review-order .subscription-details').html('p.m. inc. 30 dagen proef periode.') 

  

$('option[value="https://notificadevelo.wpengine.com/feature-requests/"]').html('Elke status')
$('.jck-sfr-filters__filter-item--category option[value="https://notificadevelo.wpengine.com/feature-requests/"]').html('Alle softwarepakketten')

$('.button.invoice').attr('target','_blank');


  
});



</script>
<!-- end Simple Custom CSS and JS -->
