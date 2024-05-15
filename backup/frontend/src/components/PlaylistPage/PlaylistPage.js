import React, { useState } from 'react';
import axios from 'axios';
import AudioPlayer from 'react-h5-audio-player';
import 'react-h5-audio-player/lib/styles.css';
import {useParams} from "react-router-dom";

const PlaylistPage = () => {

    const { emotion } = useParams();

    console.log(emotion);

    //const recognizedEmotion = props.toString();

    //console.log(recognizedEmotion.toString());
    // Dummy playlist data for demonstration
    const playlist = [
        { title: 'Song 1', url: 'song1.mp3' },
        { title: 'Song 2', url: 'song2.mp3' },
        // Add more songs as needed
    ];

    const [currentSong, setCurrentSong] = useState(null); // State to track the current song

    const playSong = (songUrl) => {
        setCurrentSong(songUrl); // Set the current song URL when a song is clicked
    };

    return (
        <div>
            <h1>Playlist for {emotion}</h1>
            <ul>
                {playlist.map((song, index) => (
                    <li key={index} onClick={() => playSong(song.url)}>
                        {song.title}
                    </li>
                ))}
            </ul>
            <div style={{ position: 'fixed', left: 0, bottom: 0, width: '100%', zIndex: 999 }}>
                <div style={{marginTop: '50px'}}>
                    <AudioPlayer
                        autoPlay
                        src={currentSong} // Pass the URL of the currently selected song
                    />
                </div>
            </div>
        </div>
    );
};

export default PlaylistPage;
