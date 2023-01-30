function myTurnToPlaceFactory(place_first, step)  {
if (place_first){
    if (step % 2 == 1) return true
} else {
    if (step % 2 == 0) return true;
}
return false;
}

module.exports = {
  myTurnToPlaceFactory
}