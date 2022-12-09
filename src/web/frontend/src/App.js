import "./App.css";
import { useState } from "react";

function App() {
  const [formValue, setFormValue] = useState("");

  const [message, setMessage] = useState([
    {
      id: 1,
      text: "Give dog a bath",
      is_user: true,
    },
    {
      id: 2,
      text: "Give dog a bath",
      is_user: false,
    },
    {
      id: 3,
      text: "Give dog a bath",
      is_user: true,
    },
  ]);
  const sendMessage = async (e) => {
    e.preventDefault();
    let copy = [...message];
    copy = [...copy, { id: copy.length + 1, text: formValue, is_user: true }];
    setMessage(copy);

    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: formValue }),
    };
    fetch("http://localhost:5000/add_input", requestOptions)
      .then((response) => response.json())
      .then((data) =>
        setMessage([
          ...copy,
          { id: data.uuid, text: data.messages, is_user: false },
        ])
      );
    setFormValue("");
  };

  return (
    <div className="App">
      <header>
        <h3>Online Predatory Conversation Detection</h3>
      </header>
      <main>
        {message &&
          message.map((msg) => <ChatMessage message={msg} key={msg.id} />)}
      </main>

      <form onSubmit={sendMessage}>
        <input
          value={formValue}
          onChange={(e) => setFormValue(e.target.value)}
          placeholder="say something nice"
        />

        <button type="submit" disabled={!formValue}>
          🕊️
        </button>
      </form>
    </div>
  );
}

function ChatMessage(props) {
  const messageClass = props.message.is_user === true ? "sent" : "received";
  const photoicon =
    props.message.is_user === true
      ? "https://static.vecteezy.com/system/resources/previews/002/318/271/original/user-profile-icon-free-vector.jpg"
      : "https://img.freepik.com/premium-vector/robot-support-bot-icon-white_116137-2172.jpg?w=2000";
  return (
    <>
      <div className={`message ${messageClass}`}>
        <img src={photoicon} alt="icon" />
        <p>{props.message.text}</p>
      </div>
    </>
  );
}

export default App;
