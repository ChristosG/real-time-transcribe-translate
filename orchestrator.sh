#!/bin/bash

# Virtual Sink: instead of recording voice via Microphone which adds noice, you can use the following 2 commands, 
# to make a Virtual Sink that captures audio directly from applications e.g. a Zoom Meeting

#pactl load-module module-null-sink sink_name=Virtual_Sink sink_properties=device.description=Virtual_Sink
#pacmd set-default-source Virtual_Sink.monitor


#Paths might need fixing
start_triton(){
        echo "Starting Triton..."
        gnome-terminal -- bash -c "cd /home/myuser/translator || exit; ./start_triton.sh; exec bash"
}

start_expo() {
    echo "Starting Expo..."
    gnome-terminal -- bash -c "cd /home/myuser/translator || exit; npx expo start; exec bash"
}

start_flask_api() {
    echo "Starting Flask API..."
    gnome-terminal -- bash -c "cd /home/myuser/translator/venv_mods || exit; uvicorn perfecto:app --host 0.0.0.0 --port 7000 --loop uvloop --workers 4; exec bash"
}


start_triton
start_expo
start_flask_api

echo "All processes have been started in separate terminals."

