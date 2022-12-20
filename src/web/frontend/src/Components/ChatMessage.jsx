import { Box, Text } from "@chakra-ui/react";

function ChatMessage(props) {
  const { id, text, is_user, time } = props.message;

  // Different styling for user and bot messages
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

export default ChatMessage;
