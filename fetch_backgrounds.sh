#!/bin/bash

set -e
mkdir -p background/misc
cd background/misc

for term in city_street dark_room dining_hall exercise_yard farm_harvest forest_growth furniture_store hair_face kitchen_baby selfie_computer_clutter selfie_crowd_cheer selfie_cute_couple selfie_dark_party selfie_face_smile selfie_graduation selfie_hold_hands selfie_museum_art selfie_point_ground selfie_street_night wall_frame; do
    term=`echo $term | sed "s/_/ /g"`
    echo $term
    pixplz --count 200 $term
done
