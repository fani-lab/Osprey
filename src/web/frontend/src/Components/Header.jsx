import { Text, VStack } from "@chakra-ui/react";

function Header() {
  return (
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
  );
}
export default Header;
