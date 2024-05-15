import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import * as faceapi from 'face-api.js';
import { Link, useHistory } from 'react-router-dom';
import styles from './Homepage.css';
import AudioPlayer from "react-h5-audio-player";

let isPaused = false;
let isNewEmotionSet = false;
let isInitialRun = true;

const Homepage = (user) => {
    const history = useHistory();
    const videoRef = useRef();
    const canvasRef = useRef();
    const [recognizedEmotion, setRecognizedEmotion] = useState('');
    const [currentEmotion, setCurrentEmotion] = useState('');
    const [selectedSongUrl, setSelectedSongUrl] = useState('');
    let mediaStream; // Define mediaStream variable
    const [currentTrack, setTrackIndex] = React.useState(null);
    const [prevEmotion, setPrevEmotion] = useState('');
    const [songs, setSongs] = useState([]);




    useEffect(() => {
        let detectionInterval;

        const loadModels = async () => {
            await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
            await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
        };

        const startVideo = () => {
            navigator.getUserMedia(
                { video: {} },
                (stream) => {
                    videoRef.current.srcObject = stream;
                    mediaStream = stream; // Assign stream to mediaStream variable
                    videoRef.current.addEventListener('loadedmetadata', () => {
                        detectEmotion();
                    });
                },
                (err) => console.error(err)
            );
        };

        console.log(user);



        console.log("This is the email : " + user.user.email);

        const detectEmotion = async () => {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            const displaySize = { width: video.videoWidth, height: video.videoHeight };
            faceapi.matchDimensions(canvas, displaySize);

            detectionInterval = setInterval(async () => {
                const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ flipHorizontal: false })).withFaceLandmarks();
                const resizedDetections = faceapi.resizeResults(detections, displaySize);

                const context = canvas.getContext('2d');
                context.clearRect(0, 0, canvas.width, canvas.height);

                resizedDetections.forEach((face) => {
                    const { x, y, width, height } = face.detection.box;
                    context.beginPath();
                    context.lineWidth = 2;
                    context.strokeStyle = 'red';
                    context.rect(x, y, width, height);
                    context.stroke();
                });

                faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);

                const snapshotCanvas = faceapi.createCanvasFromMedia(video);
                const snapshotContext = snapshotCanvas.getContext('2d');
                snapshotContext.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
                const imageBlob = await new Promise((resolve) => snapshotCanvas.toBlob(resolve));

                const formData = new FormData();
                formData.append('file', imageBlob, 'frame.png');

                try {
                    const response = await axios.post('http://localhost:6969/predict', formData, {
                        headers: {
                            'Content-Type': 'multipart/form-data'
                        }
                    });

                        console.log("Sssecon" + " " + isPaused);
                        setCurrentEmotion(response.data[0].emotion);
                        console.log(currentEmotion+"chamoooodd");
                       
                        isNewEmotionSet = false;
                        if (!isPaused){
                            const recognizedEmotion = response.data[0].emotion;
                            setRecognizedEmotion(recognizedEmotion);

                            
                            
                            if (!isInitialRun){
                                await fetchSongs(recognizedEmotion, user.user.email);
                                console.log("playlist generated on not initial run")
                            }
                            else
                            {
                                await new Promise(resolve => setTimeout(resolve, 10000));
                                console.log("waited");

                                await fetchSongs(recognizedEmotion, user.user.email);
                                console.log("playlist generated on initial run")

                            }

                            isNewEmotionSet = true;
                        }


                } catch (error) {
                    console.error('Error during prediction:', error);
                }
            }, 1000);
        };

        loadModels();
        startVideo();

        return () => {
            clearInterval(detectionInterval);
            // Stop the mediaStream when component unmounts
            if (mediaStream) {
                mediaStream.getTracks().forEach((track) => track.stop());
            }
        };
    }, []);

    useEffect(() => {
        if (selectedSongUrl !== '') {
            // Set the selected song URL to the AudioPlayer
            setSelectedSongUrl(selectedSongUrl);
        }
    }, [selectedSongUrl]);



    //Start of playlist



    useEffect(() => {
        const fetchInterval = setInterval(() => {
            console.log("Emotion", recognizedEmotion);
            fetchSongs(recognizedEmotion, user.user.email) // Default emotion, change as needed
                .catch(error => console.error('Error fetching songs:', error));
        }, 600000); // Fetch songs every hour (adjust the interval as needed)

        // Fetch songs when component mounts initially
        setRecognizedEmotion("happy");

        fetchSongs("happy", user.user.email) // Default emotion, change as needed
            .catch(error => console.error('Error fetching songs:', error));



        // Cleanup function to clear the interval when component unmounts
        return () => clearInterval(fetchInterval);

    }, []);

    const fetchSongs = async (emotion, email) => {
        try {
            const response = await axios.get(`http://localhost:6969/songlist?emotion=${emotion}&email=${email}`);
            setSongs(response.data);
            console.log(response.data);
        } catch (error) {
            console.error('Error fetching songs:', error);
        }
    };



    //End of playlist

    //New way to handle playlist


    let counter = 1; // Initialize a counter variable outside the function

    console.log(songs);
    function generatePlaylist(songs) {
        counter = 1; // Reset the counter to 1 when generating a new playlist
        const playlist = songs.map(song => ({
            id: counter++, // Assign the counter value to the id property and then increment the counter
            src: 'http://localhost:6969/songs/' +recognizedEmotion + '/' + song.Name + '.mp3',
            Name: song.Name,
            Artist: song.Artist,
            emotion: recognizedEmotion,
            value:song.value,
        }));
        return playlist;
    }

    const playlist = generatePlaylist(songs);

    console.log(playlist)

    const handleClickPlay = async (id) => {
        const isValidId = id >= 1 && id <= playlist.length; // Check if id is within the valid range
        if (isValidId) {
            console.log(id);
            setTrackIndex(id - 1); // Adjust id to match array index (id - 1)
            //isPaused = true;
            setPauseStateTrue();
           setInitialRunStateFalse()
        }
    };

    const handleEnd = async () => {
            console.log(currentEmotion+" currunt emotion ");
            console.log(recognizedEmotion +" recognized emotion");
        if(currentEmotion!== recognizedEmotion){
            setTrackIndex(0);
            console.log("heyyyy");
        }else{
            const nextTrackIndex = (currentTrack + 1) % playlist.length;
            setTrackIndex(nextTrackIndex);
        }
       
       

        setPauseStateFalse();

        // Send a request to update MongoDB database with the song name, emotion, and positive integer value
        try {
            await axios.post('http://localhost:6969/song/played', {
                Name: playlist[currentTrack].Name,
                Artist: playlist[currentTrack].Artist,
                emotion: recognizedEmotion,
                value: 100,// Positive integer value
                email: user.user.email
            });
            console.log('Song data added to database.');

        } catch (error) {
            console.error('Error adding song data to database:', error);
        }


    };

    function handlePlay() {

        setInitialRunStateFalse()

        //isPaused = true;

        if (isNewEmotionSet){
            setPauseStateTrue();

        }

        // Additional actions you want to perform when audio starts playing
    };

    // Function to play next song
    function handleNext() {

        if(currentEmotion!== recognizedEmotion){
            setTrackIndex(0);
            console.log("heyyyy");
        }else{
            const nextTrackIndex = (currentTrack + 1) % playlist.length;
            setTrackIndex(nextTrackIndex);
        }
        
        setPauseStateFalse();
        console.log("This is on handleNext " + isPaused);

        // Send a request to update MongoDB database with the song name, emotion, and positive integer value
        axios.post('http://localhost:6969/song/played', {
            Name: playlist[currentTrack].Name,
            Artist: playlist[currentTrack].Artist,
            emotion: recognizedEmotion,
            value: -100, // Negative integer value
            email: user.user.email
        })
            .then(() => {
                console.log('Song data added to database.');
            })
            .catch(error => {
                console.error('Error adding song data to database:', error);
            });
    }


    // Function to play previous song
    const handlePrev = () => {
        const prevTrackIndex = (currentTrack - 1 + playlist.length) % playlist.length;
        setTrackIndex(prevTrackIndex);
    };

    function setPauseStateTrue(){

        isPaused = true;

    }

    function setPauseStateFalse(){

        isPaused = false;

    }

    function setInitialRunStateFalse(){

        isInitialRun = false;
    }



    //End of new way to handle playlist

    const handleMouseEnter = (event) => {
        event.target.style.cursor = 'pointer';
    };

    const handleMouseLeave = (event) => {
        event.target.style.cursor = 'auto';
    };

    return (
        <div className="container">
            <div className="webcam-column">
                <div style={{position: 'relative', width: '100%', height: '400px'}}>
                    <video
                        ref={videoRef}
                        style={{
                            position: 'absolute',
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                            transform: 'scaleX(-1)'
                        }}
                        autoPlay
                        muted
                    />
                    <canvas
                        ref={canvasRef}
                        style={{
                            position: 'absolute',
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                            transform: 'scaleX(-1)'
                        }}
                    />
                    
                </div>
                <button onClick={setPauseStateFalse} className="reset-button">
                 <i className="fas fa-sync-alt"></i> Reset
                </button>
                <div>Currently playing based on emotion: {recognizedEmotion}</div>
                <div>Current emotion: {currentEmotion}</div>
                <br></br>
                <br></br>
                
   

            </div>

            <div className="songlist-container"> {/* Container with the fixed height and scrollbar */}
            <div className={styles['song-list']}>
                {playlist.map((song) => (
                    <div
                        key={song.id}
                        className={`song-card card-${song.id % 3 + 1}`}
                        onClick={() => handleClickPlay(song.id)}
                        onMouseEnter={handleMouseEnter}
                        onMouseLeave={handleMouseLeave}
                    >
                        <h5>{song.Name}</h5>
                        <p>{song.Artist}</p>
                        <p>User Rating : {song.value}</p>
                    </div>
                ))}
            </div>
        </div>
            <div style={{position: 'fixed', left: 0, bottom: 0, width: '100%', zIndex: 999}}>

                <div style={{marginTop: '50px'}}>
                    {currentTrack !== null && (<p>Now Playing : {playlist[currentTrack].Name} </p>)}
                    {currentTrack !== null && (
                        <AudioPlayer
                            volume="0.5"
                            src={playlist[currentTrack].src}
                            showSkipControls
                            onPlay={handlePlay}
                            onEnded={handleEnd}
                            onClickNext={handleNext}
                            onClickPrevious={handlePrev}
                            onError={() => console.log('play error')}
                            // Try other props!
                        />
                    )}
                </div>
            </div>

        </div>
    );


};

export default Homepage;
