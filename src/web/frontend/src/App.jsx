import "./App.css";
import { useState, useRef, useEffect } from "react";
import { AiOutlineSend } from "react-icons/ai";
import ChatMessage from "./Components/ChatMessage";
import {
  Button,
  Box,
  Input,
  useToast,
  InputGroup,
  InputRightElement,
} from "@chakra-ui/react";
import Header from "./Components/Header";

function App() {
  const dummy = useRef();
  let date = new Date();
  const toast = useToast();
  const id = "toastId";

  const [formValue, setFormValue] = useState("");
  const [message, setMessage] = useState([]);
  const [isLoad, setIsLoading] = useState(false);
  const [displayToast, setDisplayToast] = useState(false);
  const [toastMsg, setToastMsg] = useState("");

  // Autoscroll to bottom of chat everytime a new message is sent
  useEffect(() => {
    scrollToBottom();
  }, [message]);

  const scrollToBottom = () => {
    dummy.current?.scrollIntoView({ behavior: "smooth" });
  };

  // After user sends a message
  const sendMessage = async (e) => {
    e.preventDefault();
    setIsLoading(!isLoad);

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
    setFormValue("");
    dummy.current.scrollIntoView({ behavior: "smooth" });
    console.log(date.valueOf());
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: formValue,
        //no date, just time
        time: date.getHours() + "." + date.getMinutes(),
      }),
    };

    // Checks if user message is predatorial
    fetch("http://localhost:5000/classify_user_msg", requestOptions).then(
      (response) =>
        response.json().then((data) => callToast(data.user_label, formValue))
    );

    // Receieves response from chatbot
    // After receiving response, user can send another message
    await fetch("http://localhost:5000/add_input", requestOptions)
      .then((response) => response.json())
      .then(
        (data) => (
          setMessage([
            ...copy,
            {
              id: copy.length + 1,
              text: data.messages,
              is_user: false,
              time: date.getHours() + ":" + date.getMinutes(),
            },
          ]),
          callToast(data.chatbot_label, data.messages)
        )
      );
    setIsLoading(false);
  };

  // Resets and clears conversation
  function reset() {
    setMessage([]);
    fetch("http://localhost:5000/reset"); // Chatbot forgets past conversation
  }

  // Displays toast if message is predatorial
  function callToast(result, msg) {
    if (result === "1") {
      setDisplayToast(true);
      setToastMsg(msg);
    }

    // Clear toast
    setTimeout(() => {
      setDisplayToast(false);
      setToastMsg("");
    }, 5000);
  }

  return (
    <Box className="App">
      {displayToast && !toast.isActive(id) // Prevents multiple toasts from being displayed
        ? toast({
            title: "Predatorial message detected.",
            description: `${toastMsg}`,
            status: "error",
            id,
            position: "top",
            isClosable: true,
          })
        : null}
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
                isLoading={isLoad}
                disabled={!formValue || isLoad}
              >
                <AiOutlineSend />
              </Button>
            </InputRightElement>
          </InputGroup>
        </form>
        <Box className="footer">
          {/*<Text>
            Research project by{" "}
            <Link
              color="teal.500"
              href="https://github.com/fani-lab/online_predatory_conversation_detection"
            >
              Fani's lab
            </Link>
        </Text> */}

          <Button
            disabled={isLoad || message.length === 0}
            colorScheme={"yellow"}
            onClick={reset}
          >
            Reset
          </Button>
        </Box>
      </Box>
    </Box>
  );
}

export default App;
