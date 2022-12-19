import "./App.css";
import { useState, useRef, useEffect } from "react";
import { AiOutlineSend } from "react-icons/ai";
import ChatMessage from "./Components/ChatMessage";
import {
  Button,
  Box,
  Text,
  Input,
  Link,
  InputGroup,
  InputRightElement,
} from "@chakra-ui/react";
import Header from "./Components/Header";

function App() {
  const dummy = useRef();
  let date = new Date();

  const [formValue, setFormValue] = useState("");
  const [message, setMessage] = useState([]);

  useEffect(() => {
    scrollToBottom();
  }, [message]);

  // Autoscroll to bottom of chat everytime a new message is sent
  const scrollToBottom = () => {
    dummy.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Sends message to backend and receives response
  const sendMessage = async (e) => {
    e.preventDefault();
    let copy = [...message];
    copy = [
      ...copy,
      {
        id: copy.length + 1,
        text: formValue,
        is_user: true,
        time: date.getHours() + ":" + date.getMinutes(),
      },
    ];
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
          {
            id: copy.length + 1,
            text: data.messages,
            is_user: false,
            time: date.getHours() + ":" + date.getMinutes(),
          },
        ])
      );
    setFormValue("");
    dummy.current.scrollIntoView({ behavior: "smooth" });
  };

  // Resets and clears conversation
  function reset() {
    setMessage([]);
    fetch("http://localhost:5000/reset");
  }

  return (
    <Box className="App">
      <Header />
      <main>
        {message &&
          message.map((msg) => <ChatMessage message={msg} key={msg.id} />)}
        <span ref={dummy}></span>
      </main>
      <br />
      <Box className="Bottom">
        <form onSubmit={sendMessage}>
          <InputGroup size="lg">
            <Input
              value={formValue}
              onChange={(e) => setFormValue(e.target.value)}
              placeholder="Type your message"
            />
            <InputRightElement>
              <Button
                size="md"
                variant="ghost"
                type="submit"
                disabled={!formValue}
              >
                <AiOutlineSend />
              </Button>
            </InputRightElement>
          </InputGroup>
        </form>
        <Box className="footer">
          <Text>
            Research project by{" "}
            <Link
              color="teal.500"
              href="https://github.com/fani-lab/online_predatory_conversation_detection"
            >
              Fani's lab
            </Link>
          </Text>

          <Button colorScheme={"yellow"} onClick={reset}>
            Reset
          </Button>
        </Box>
      </Box>
    </Box>
  );
}

export default App;
