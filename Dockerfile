FROM tensorflow/tensorflow:2.15.0.post1-gpu

RUN mkdir /app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    qt5-gtk-platformtheme \
    libqt5gui5 \
    libqt5dbus5 \
    qtbase5-dev \
    qttools5-dev-tools


RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install ultralytics opencv-python

COPY . /app/

ENTRYPOINT [ "python3", "./main.py" ]
CMD [ "--source", "", "--output", "", "--caption_url", "", "--port", "" ]