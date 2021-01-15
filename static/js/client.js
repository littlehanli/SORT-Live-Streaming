
function fetch_from_server(){
    var req = new XMLHttpRequest();
    req.open('GET', '/tracking_list');
    req.onreadystatechange = function () {
      if (req.readyState === 4) {
        //console.log(JSON.parse(req.responseText));
        process_option_list(req.responseText);
      }
    };
    req.send();
}
function process_option_list(v){
    
    var pastArray = $.map($('#tracking_id option'), function(e) { 
        if(e.value != "All" && e.value!='None'){
            return parseInt(e.value);
        }
        else return e.value
    });
    selectArray = JSON.parse(v)
    //console.log(pastArray)
    // elements add : selectArray - pastArray
    var add = selectArray.filter(function(x) { return pastArray.indexOf(x) < 0 })
    //console.log(add)

    // elements remove : pastArray - selectArray
    var remove = pastArray.filter(function(x) { return selectArray.indexOf(x) < 0 })
    //console.log(remove)

    var selectBox = document.getElementById('tracking_id');
    //selectBox.options.length = 0;  // remove all
    for(var i = 0; i < add.length; i++){
        selectBox.options.add(new Option(add[i], add[i]));
    }
    
    for (var i = 0; i < remove.length; i++){
        for (var j=0; j<selectBox.length; j++) {
            if (selectBox.options[j].value == remove[i])
                selectBox.remove(j);
        }
    }
}

function repeatedly_get(){
    fetch_from_server();
    setTimeout(repeatedly_get, 5000);
}


var pickID = 'All';
/////////////////main//////////////////
$(document).ready(function(){  

    repeatedly_get();
    
    var selectBox = document.getElementById('tracking_id');
    var btn = document.getElementById('select_id')
    
    
    //ajax
    btn.addEventListener('click',function(event){
        event.preventDefault();

        pickID = selectBox.value
        console.log(pickID)

        var data = new FormData();
        data.append('id',pickID);
        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/get_select_id', true);
        
        xhr.send(data);

        xhr.addEventListener('loadend', function(){
            if(xhr.status == 201){
                var res_json = JSON.parse(xhr.responseText);
                document.getElementById('pick').innerHTML = res_json.id;
            }
        }, false);
    }, false);
})