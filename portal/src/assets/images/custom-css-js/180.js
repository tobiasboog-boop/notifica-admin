<!-- start Simple Custom CSS and JS -->
<script type="text/javascript">
jQuery(document).ready(function( $ ){
    jQuery.validator.setDefaults({
    debug: false,
    success: "valid"
  });
var language = $('html').attr('lang');
var customPhoneMessage = "Please enter a valid phone number";
var customAddressMessage = "Vul een geldig adres in"
translateValidationMessages(language)  
    
//Vul bij deze onderstaande rules de 'name' in van het veld dat je wilt valideren, deze staat in <input>. Zet deze altijd tussen ""
    
//Wil je dat een veld niet verplicht is voor de gebruiker? Zet dan een comment voor required bij het bijbehorende veld. 
//----UITZONDERING!---- Wil je dat het VERPLICHT is om een telefoonnummer achter te laten? Ga dan naar de TELEFOONUMMER FUNCTIE en zet de return op false.
//Deze staat standaard op NIET VERPLICHT!
    
//Wil je een veld toevoegen die niet standaard in de lijst staat? Zoek hieronder het bijbehorende veld en haal alle rijen comments weg. Met andere woorden de regels met { tot en met },

//Heb je een site waar meerdere formulieren in staan die niets te maken hebben met validatie? Dan moet de onderstaande eerste regel aangepast worden. 
//bijvoorbeeld $("#gform_1, #gform_2"), afhankelijk hoeveel talen je er in hebt zitten. Denk aan de komma tussen de ID's. De taal ontdekt het script vanzelf, hoef je niets voor te doen.

//-------------------- FORMULIER VALIDATIE ------------------- 
  $("form[id^='gform'").validate({ // VUL HIER DE NAMES IN VAN DE VELDEN
          rules: {
               "input_6": { // voornaam
                   required: true,
                   letterswithbasicpunc: true
                  },
               "input_1.6": { // Achternaam
                  required: true,
                  letterswithbasicpunc: true
                  },          
               "input_4": {  // Email
                  required: true,
                  email: true
                  },
               "input_3": {  // Email
                  required: true,
                  email: true
                  },
               "input_5": { // Bericht
                   required: true              
                  },
               "input_2": { // Telefoonnummer
 				   dutchPhone: true
                  },            
  //           "input_3": { // Postcode
  //                 required: true,
  //               	postalcodeNL: true,
  //                 minlength: 6
  //             },
  //           "input_4": { // Huisnummer
  //                 required: true,
  //               	number: true
  //             },
  //           "input_5": { // Toevoeging
  //                 required: false
  //             },

          }
      },
);
  $('input[id^="gform_submit_button_"').on('click', function(){
    if($("form[id^='gform'").valid()){
        $("form[id^='gform'").submit();
    }
});
//-------------------- CHECKOUT VALIDATIE ------------------- 
  $("form[name='checkout'").validate({ // VUL HIER DE NAMES IN VAN DE VELDEN
          rules: {
               "billing_first_name": { // voornaam
                   required: true,
                   letterswithbasicpunc: true
                  },
               "billing_last_name": { // Achternaam
                  required: true,
                  letterswithbasicpunc: true
                  },   
               "billing_company": { // Adres
                  required: true
                  }, 
               "billing_address_1": { // Adres
                  adres: true
                  },  
               "billing_email": {  // Email
                  required: true,
                  email: true
                  },
               "billing_phone": { // Telefoonnummer
 				   dutchPhone: true,
                   required:true
                  },            
               "billing_postcode": { // Postcode
                   required: true,
                   postalcodeNL: true,
  	               minlength: 6
  	              },
               "billing_city": { // Plaats
                   required: true,
                   letterswithbasicpunc: true
                  },
               "billing_account_info": { // Plaats
                   required: true
                  },
               "account_username": { // Plaats
                   required: true
                  },
               "account_password": { // Plaats
                   required: true
                  },
               "shipping_first_name": { // voornaam
                   required: true,
                   letterswithbasicpunc: true
                  },
               "shipping_last_name": { // Achternaam
                  required: true,
                  letterswithbasicpunc: true
                  },      
               "shipping_address_1": { // Adres
                  adres: true
                  },            
               "shipping_postcode": { // Postcode
                   required: true,
                   postalcodeNL: true,
  	               minlength: 6
  	              },
               "shipping_city": { // Plaats
                   required: true,
                   letterswithbasicpunc: true
                  },
  //           "input_4": { // Huisnummer
  //                 required: true,
  //               	number: true
  //             },
  //           "input_5": { // Toevoeging
  //                 required: false
  //             },

          }
      },
);
  
// ------ TELEFOONNUMMER FUNCTIE ----  
  
  jQuery.validator.addMethod("dutchPhone", function (value, element) {
     if (language =='nl-NL' || language =='nl'){
	customMessage = "Vul een geldig telefoonnummer in."   
	} 
    value = value.replace(/\s+/g,"").replace(/\-+/g,"").replace(/\.+/g,"")    
    if(value == ''){
//Verander onderstaande true naar false als je wilt dat een telefoonnummer VERPLICHT is.
      return true;
    }
    if(/^\(\+[0-9]{2}\)\(0\)\d{9}$/i.test(value)){
    return true;
      }
    if(/^\+[0-9]{2}\(0\)\d{9}$/i.test(value)){
    return true;
      }  
    if(/^\(\+[0-9]{2}\)\d{9}$/i.test(value)){
    return true;
      }    
    if(/^\+[0-9]{2}\d{9}$/i.test(value)){
    return true;
    }
    if(/^\([0-9]{3,4}\)\d{6}$/i.test(value)){
    return true;
      }
    if (/^(\d{10},)*\d{10}$/.test(value)){
    return true
    }
     else {
        return false;
    };

  }, customPhoneMessage);
  
//------- ADRES VALIDATIE ------------//
    jQuery.validator.addMethod("adres", function (value, element){
    if (language =='nl-NL' || language =='nl'){
	customMessage = "Vul een geldig adres in."   
	} 
    if(/^[\w\s.-]+\d+/g.test(value)){
    return true;
      } 
    }, customAddressMessage)
  
  
  
  
  function translateValidationMessages(language) {
  message = {
    nl: {
      required: "Vul dit veld in.",
      letterswithbasicpunc: "Vul een geldige naam in.",
      email: "Vul een geldig e-mail adres in",
      url: "Plaats een geldige URL.",
      date: "Vul een geldige datum in.",
      number: "Vul een geldig telefoonnummer in.",
      digits: "Vul alleen getallen in.",
      creditcard: "Vul een geldig rekeningnummer in.",
	  dutchPhone: "Vul een geldig telefoonnummer in.",
      adres: "Vul een geldig adres in"
    },
  };
  if (language =='nl-NL' || language =='nl') {
    $.extend($.validator.messages, message.nl);
    customPhoneMessage = "Vul een geldig telefoonnummer in.";
  } 
}
  });
    
  </script>
<!-- end Simple Custom CSS and JS -->
