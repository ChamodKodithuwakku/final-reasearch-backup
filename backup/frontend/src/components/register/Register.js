import React, { useState } from 'react';
import axios from 'axios';
import Login from '../login/Login'
import {Link} from "react-router-dom";

const Register = () => {
    const [user, setUser] = useState({
        name: '',
        email: '',
        password: '',
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setUser({
            ...user,
            [name]: value,
        });
    };

    // Fix typo in function name "egister" to "register"
    const register = () => {
        const { name, email, password } = user;
        if (name && email && password) {
            axios.post('http://localhost:6969/Register', user)
                .then((res) => console.log(res))
                .catch((error) => console.error(error)); // Add error handling
        } else {
            alert('invalid input');
        }
    };

    return (
        <>
            <div className="flex flex-col max-w-md px-4 py-8 bg-white rounded-lg shadow dark:bg-gray-800 sm:px-6 md:px-8 lg:px-10">
                <div className="self-center mb-2 text-xl font-light text-gray-800 sm:text-2xl dark:text-white">
                    Create a new account
                </div>
                <span className="justify-center text-sm text-center text-gray-500 flex-items-center dark:text-gray-400">
          Already have an account ?
          <Link to="/login" className="text-sm text-blue-500 underline hover:text-blue-700">
                    Sign in
                </Link>
        </span>
                <div className="p-6 mt-8">
                    <form action="#">
                        <div className="flex flex-col mb-2">
                            <div className=" relative">
                                <input
                                    type="text"
                                    id="create-account-pseudo"
                                    className="rounded-lg border-transparent flex-1 appearance-none border border-gray-300 w-full py-2 px-4 bg-white text-gray-700 placeholder-gray-400 shadow-sm text-base focus:outline-none focus:ring-2 focus:ring-purple-600 focus:border-transparent"
                                    name="name"
                                    value={user.name}
                                    onChange={handleChange}
                                    placeholder="FullName"
                                />
                            </div>
                        </div>
                        <div className="flex gap-4 mb-2">
                            <div className=" relative">
                                <input
                                    type="text"
                                    id="create-account-first-name"
                                    className="rounded-lg border-transparent flex-1 appearance-none border border-gray-300 w-full py-2 px-4 bg-white text-gray-700 placeholder-gray-400 shadow-sm text-base focus:outline-none focus:ring-2 focus:ring-purple-600 focus:border-transparent"
                                    name="email"
                                    value={user.email}
                                    onChange={handleChange}
                                    placeholder="Email"
                                />
                            </div>
                        </div>
                        <div className="flex flex-col mb-2">
                            <div className=" relative">
                                <input
                                    type="password"
                                    id="create-account-email"
                                    className="rounded-lg border-transparent flex-1 appearance-none border border-gray-300 w-full py-2 px-4 bg-white text-gray-700 placeholder-gray-400 shadow-sm text-base focus:outline-none focus:ring-2 focus:ring-purple-600 focus:border-transparent"
                                    name="password"
                                    value={user.password}
                                    onChange={handleChange}
                                    placeholder="Password"
                                />
                            </div>
                        </div>
                        <div className="flex w-full my-4">
                            <button
                                type="button" // Change type to button
                                className="py-2 px-4 bg-purple-600 hover:bg-purple-700 focus:ring-purple-500 focus:ring-offset-purple-200 text-white w-full transition ease-in duration-200 text-center text-base font-semibold shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2 rounded-lg"
                                onClick={register}
                            >
                                Register
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </>
    );
};

export default Register;
