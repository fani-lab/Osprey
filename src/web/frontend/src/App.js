import "./App.css";
import { useState, useRef, useEffect } from "react";
import { AiOutlineSend } from "react-icons/ai";
import {
  Button,
  Box,
  Heading,
  Text,
  Input,
  VStack,
  HStack,
  Link,
  InputGroup,
  InputRightElement,
} from "@chakra-ui/react";
function App() {
  const dummy = useRef();
  let date = new Date();

  const [formValue, setFormValue] = useState("");

  const [message, setMessage] = useState([]);

  useEffect(() => {
    scrollToBottom();
  }, [message]);

  const scrollToBottom = () => {
    dummy.current?.scrollIntoView({ behavior: "smooth" });
  };
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

  function reset() {
    setMessage([]);
    fetch("http://localhost:5000/reset");
  }

  return (
    <Box className="App">
      <header>
        <VStack>
          <Text fontSize="3xl" fontWeight="semibold">
            Online Predatory Conversation Detection
          </Text>
          {/*<Text>
            Talk to a chat bot! You’ll be notified if a machine learning
            algorithm detects a predatory conversation.{" "}
  </Text>*/}
        </VStack>
      </header>
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

function ChatMessage(props) {
  const { id, text, is_user, time } = props.message;

  const messageClass = is_user === true ? "sent" : "received";
  const leftOrRight = is_user === true ? "right" : "left";

  return (
    <>
      <Box style={{ display: "flex", flexDirection: "column" }}>
        <Box style={{ textAlign: `${leftOrRight}` }}>Today at {time}</Box>
        <Box className={`message ${messageClass}`}>
          <Text fontSize="xl">{text}</Text>
        </Box>
      </Box>
    </>
  );
}

export default App;
