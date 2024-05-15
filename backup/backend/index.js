import express from "express";
import cors from "cors";
import mongoose from "mongoose";
import { fileURLToPath } from 'url';
import path, { dirname } from 'path';
import * as tf from '@tensorflow/tfjs-node';
import multer from 'multer';
import { createRequire } from 'module';
import * as csv from 'csv-parse';

import cv from 'opencv4nodejs'

import { promises as fsPromises } from 'fs';
import bodyParser from "body-parser";  // Use `promises` to read the file asynchronously
//const faceUtil = require('/lib/face');
import * as faceUtil from './lib/face.js';
//const imageUtil = createRequire('/lib/image');
import * as imageUtil from './lib/image.js';
import _ from "lodash";
import axios from "lodash";
import * as fs from "fs";

//import util from "./lib/util.js";


const storage = multer.memoryStorage();
const upload = multer({ storage: storage });


const { readFile } = fsPromises;

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(cors());
app.use('/mp3', express.static('mp3'));


mongoose.connect("mongodb://localhost:27017/auth", {
    useNewUrlParser: true,
    useUnifiedTopology: true
}, () => {
    console.log("connected to DB");
});


app.use(bodyParser.json({ limit: '10mb' })); // Increase payload limit if needed



//user schema
const userSchema = new mongoose.Schema({
    name: String,
    email: String,
    password: String
})

const User = new mongoose.model("User", userSchema)

//routes routes
app.post("/Login",(req,res)=>{
    const {email,password} =req.body;
    User.findOne({email:email},(err, user)=>{
        if(user){
           if(password === user.password){
               res.send({message:"login sucess",user:user})
           }else{
               res.send({message:"wrong credentials"})
           }
        }else{
            res.send("not register")
        }
    })
});
app.post("/Register",(req,res)=>{
    console.log(req.body)
    const {name,email,password} =req.body;
    User.findOne({email:email},(err,user)=>{
        if(user){
            res.send({message:"user already exist"})
        }else {
            const user = new User({name,email,password})
            user.save(err=>{
                if(err){
                    res.send(err)
                }else{
                    res.send({message:"sucessfull"})
                }
            })
        }
    })


})

const modelPath = 'models/models/model.json';

// Load the TensorFlow.js model asynchronously
async function loadModel() {
    //const modelPath = 'models/emotion_model.h5';


    try {
        // Load the model using tf.loadLayersModel
        const model = await tf.loadLayersModel(`file://${modelPath}`);
        console.log('Model loaded successfully');
        return model;
    } catch (error) {
        console.error('Error loading the model:', error);
        throw error;
    }
}
// Function to preprocess image data and obtain the TensorFlow tensor

//const model = await loadModel();

// API endpoint for emotion prediction
// Define a route for emotion prediction
app.post('/predict', upload.single('file'),async (req, res) => {

    try {
        //console.log(req);
        let image = req.file.buffer;

        if (!image) {
            return res.status(400).json({ error: 'No file' });
        }

        console.log('Received image_data:', image);

        const emotionModel = await tf.loadLayersModel(`file://${modelPath}`);
       let color = 'black';

        if (!color) {
            color = 'black';
        }

        const colorVec = imageUtil.getColorVecByString(color);

        const inputShape = [
            emotionModel.feedInputShapes[0][1],
            emotionModel.feedInputShapes[0][2],
        ];

        // Fetch image from the API
        const response = await axios.get(req.file, { responseType: 'arraybuffer' });
        const imageData = Buffer.from(image, 'binary');
        const imageRGB = cv.imdecode(imageData);


        const imageGray = imageRGB.cvtColor(cv.COLOR_BGR2GRAY);


        const faces = await faceUtil.getFaces(imageGray);

        console.log(faces)



        const results = [];

        for (const face of faces) {
            const x = new cv.Point2(face.x, face.y);
            const y = new cv.Point2(face.x + face.width, face.y + face.height);
            imageRGB.drawRectangle(x, y, colorVec);

            const faceImage = imageRGB.getRegion(face);

            console.log(faceImage)
            const tensor = await faceUtil.preprocessToTensor(faceImage, inputShape);

            console.log(tensor);

            const emotion = await faceUtil.inferEmotion(tensor, emotionModel);

            console.log(emotion);
            results.push({ face: { x: face.x, y: face.y, height: face.height, width: face.width }, emotion });
        }

        res.json(results);
        console.log(results);
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Internal server error' });
    }
});



async function preprocessToTensor (faceImage, targetSize) {
    faceImage = await faceImage.resizeAsync(targetSize[0], targetSize[1])
    faceImage = await faceImage.bgrToGrayAsync()

    let tensor = tf.tensor3d(_.flattenDeep(faceImage.getDataAsArray()), [64, 64, 1])

    tensor = tensor.asType('float32')
    tensor = tensor.div(255.0)
    tensor = tensor.sub(0.5)
    tensor = tensor.mul(2.0)
    tensor = tensor.reshape([1, 64, 64, 1])

    return tensor
}

async function preprocessImage(imageBuffer, inputShape) {

    let tensor = tf.node.decodeImage(imageBuffer, 1);
    tensor = tensor.resizeBilinear(inputShape[0], inputShape[1])
    tensor = tf.tensor3d(_.flattenDeep(tensor), [64, 64, 1])
    tensor = tensor.asType('float32')
    tensor = tensor.div(255.0)
    tensor = tensor.sub(0.5)
    tensor = tensor.mul(2.0)
    tensor = tensor.reshape([1, 64, 64, 1])

    return tensor;
}

async function inferEmotion (tensor, model) {
    const result = await model.predict(tensor)

    return getEmotionLabel(result)
}

const EMOTION_LABELS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

function getEmotionLabel (result) {
    const labels = Array.from(result.dataSync()).slice(0, 7)

    let maxIdx = null
    let maxVal = null
    for (let i = 0; i < labels.length; i++) {
        const val = labels[i]
        if (maxVal === null || maxVal < val) {
            maxVal = val
            maxIdx = i
        }
    }

    return EMOTION_LABELS[maxIdx]
}

//Start of song suggestion

const dataDir = path.join(__dirname, 'songs');


app.get('/songlist', async (req, res) => {
    const { emotion, email } = req.query;
    if (!emotion) {
        return res.status(400).json({ error: 'Emotion parameter is required.' });
    }
    try {
        const songsWithEmotion = await Song.find({ emotion, email, value: { $gte: 0 } })
            .sort({ value: -1 })
            .select('Name Artist emotion value');

        console.log("There are " + songsWithEmotion.length + " songs with the specified emotion.");

        let remainingCount = 10 - songsWithEmotion.length;
        if (remainingCount < 0) {
            remainingCount = 0; // Ensure remaining count is not negative
        }

        const filePath = path.join(dataDir, `${emotion}.csv`);
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ error: 'Emotion file not found.' });
        }

        readCSVFile(filePath, (songsFromCSV) => {
            //this is the original code line
            //const randomSongs = getRandomSongs(songsFromCSV, remainingCount);

            const filteredSongsFromCSV = songsFromCSV.filter(song => !songsWithEmotion.some(s => s.Name === song.Name));

            const randomSongs = getRandomSongs(filteredSongsFromCSV, remainingCount);

            // Combine both sets of songs and send as the response
            const combinedSongs = [...songsWithEmotion, ...randomSongs];
            res.json(combinedSongs);
        });

    } catch (error) {
        console.error('Error fetching songs from database:', error);
        res.status(500).json({ error: 'An error occurred while fetching songs from database.' });
    }
});

// Function to read a CSV file
const readCSVFile = (filePath, callback) => {
    const fileData = fs.readFileSync(filePath, 'utf8');
    csv.parse(fileData, { columns: true }, (err, data) => {
        if (err) {
            console.error('Error parsing CSV:', err);
            callback([]);
        } else {
            callback(data);
        }
    });
};

// Function to select random songs
const getRandomSongs = (songs, count) => {
    const shuffledSongs = songs.sort(() => 0.5 - Math.random());
    return shuffledSongs.slice(0, count);
};
//End of song Suggestion



// Start of mp3 file retrieval
const mp3Directory = path.join(__dirname, 'mp3');

// Serve MP3 files statically
app.use('/songs', express.static(mp3Directory));

// End of mp3 file retrieval


//Start of Play algorithm


// Define Schema for song data
const songSchema = new mongoose.Schema({
    Name: String,
    Artist: String,
    emotion: String,
    value: Number,
    email: String
});
// Define Model
const Song = mongoose.model('Song', songSchema);

// Route to handle adding or updating song data
app.post('/song/played', async (req, res) => {
    try {
        // Extract data from request body
        const { Name, Artist, emotion, value, email } = req.body;

        // Check if a song with the same name exists in the database
        const existingSong = await Song.findOne({ Name, email });

        if (existingSong) {
            // If the song exists, update its value by incrementing
            existingSong.value += value;
            await existingSong.save();
            res.status(200).json({ message: 'Song data updated in database.' });
        } else {
            // If the song doesn't exist, create a new entry
            const newSong = new Song({
                Name,
                Artist,
                emotion,
                value,
                email
            });
            await newSong.save();
            res.status(201).json({ message: 'Song data added to database.' });
        }
    } catch (error) {
        console.error('Error adding/updating song data to database:', error);
        res.status(500).json({ error: 'An error occurred while adding/updating song data to database.' });
    }
});

//End


app.listen(6969, () => {
    console.log("Server started on port 6969");
});