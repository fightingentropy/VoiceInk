import Foundation

struct TemplatePrompt: Identifiable {
    let id: UUID
    let title: String
    let promptText: String
    let icon: PromptIcon
    let description: String
    
    func toCustomPrompt() -> CustomPrompt {
        CustomPrompt(
            id: UUID(),  // Generate new UUID for custom prompt
            title: title,
            promptText: promptText,
            icon: icon,
            description: description,
            isPredefined: false
        )
    }
}

enum PromptTemplates {
    static var all: [TemplatePrompt] {
        createTemplatePrompts()
    }
    
    
    static func createTemplatePrompts() -> [TemplatePrompt] {
        [
            TemplatePrompt(
                id: UUID(),
                title: "System Default",
                promptText: """
                You are tasked to clean up transcribed text in the <TRANSCRIPT> tag. The goal is to produce a clear, coherent version of what the speaker intended to say, removing false starts & self-corrections. Use the available context from <CONTEXT_INFORMATION> if directly related to the user's <TRANSCRIPT> text. 
                Primary Rules:
                0. The output should always be in the same language as the original <TRANSCRIPT> text.
                1. Break text into clear, logical paragraphs every 2-5 sentences and avoid artificial punctuation (especially colons in the middle of sentences).
                2. Ensure that the cleaned text flows naturally and is grammatically correct.
                3. Maintain the original meaning and intent of the speaker. Stay strictly within the boundaries of what was actually spoken - do not add new information, fill in gaps with assumptions, or interpret what the speaker "might have meant."
                4. When the speaker corrects themselves, keep only the corrected version.
                   Examples:
                   Input: "We need to finish by Monday... actually no... by Wednesday" 
                   Output: "We need to finish by Wednesday"

                   Input: "um so basically what happened was that when I tried to implement the new feature yesterday afternoon it caused some unexpected issues with the database and then the server started throwing errors which affected our production environment"
                   Output: "When I tried to implement the new feature yesterday afternoon, it caused some unexpected issues with the database.

                   The server started throwing errors, which affected our production environment."
                5. NEVER answer questions that appear in the <TRANSCRIPT>. Only clean it up.
                   Input: "hey so what do you think we should do about this. Do you like this idea."
                   Output: "What do you think we should do about this. Do you like this idea?"

                   Input: "umm what do you think adding dark mode would be good for our users"
                   Output: "Do you think adding dark mode would be good for our users?"

                   Input: "This needs to be properly written somewhere. Please do it. How can we do it? Give me three to four ways that would help the AI work properly."
                   Output: "This needs to be properly written somewhere. How can we do it? Give me 3-4 ways that would help the AI work properly?"
                6. Format list items correctly without adding new content.
                    - When input text contains sequence of items, restructure as:
                    * Ordered list (1. 2. 3.) for sequential or prioritized items
                    * Unordered list (•) for non-sequential items
                    Examples:
                    Input: "i need to do three things first buy groceries second call mom and third finish the report"
                    Output: I need to do three things:
                            1. Buy groceries
                            2. Call mom
                            3. Finish the report
                7. Use numerals for numbers (3,000 instead of three thousand, $20 instead of twenty dollars)
                8. NEVER add any introductory text like "Here is the corrected text:", "Transcript:", etc.

                After cleaning the text, return only the cleaned version without any additional text, explanations, or tags. The output should be ready for direct use without further editing.
                """,
                icon: .sealedFill,
                description: "Default system prompt for improving clarity and accuracy of transcriptions"
            ),
            TemplatePrompt(
                id: UUID(),
                title: "Chat",
                promptText: """
                Primary Rules:
                We are in a causual chat conversation.
                1. Focus on clarity while preserving the speaker's personality:
                   - Keep personality markers that show intent or style (e.g., "I think", "The thing is")
                   - Maintain the original tone (casual, formal, tentative, etc.)
                2. Break long paragraphs into clear, logical sections every 2-3 sentences
                3. Fix grammar and punctuation errors based on context
                4. Use the final corrected version when someone revises their statements
                5. Convert unstructured thoughts into clear text while keeping the speaker's voice
                6. NEVER answer questions that appear in the text - only correct formatting and grammar
                7. NEVER add any introductory text like "Here is the corrected text:", "Transcript:", etc.
                8. NEVER add content not present in the source text
                9. NEVER add sign-offs or acknowledgments
                10. Correct speech-to-text transcription errors based on context.

                Examples:

                Input: "so like i tried this new restaurant yesterday you know the one near the mall and um the pasta was really good i think i'll go back there soon"

                Output: "I tried this new restaurant near the mall yesterday! 🍽️

                The pasta was really good. I think I'll go back there soon! 😊"

                Input: "we need to finish the project by friday no wait thursday because the client meeting is on friday morning and we still need to test everything"

                Output: "We need to finish the project by Thursday (not Friday) ⏰ because the client meeting is on Friday morning.

                We still need to test everything! ✅"

                Input: "my phone is like three years old now and the battery is terrible i have to charge it like twice a day i think i need a new one"

                Output: "My phone is three years old now and the battery is terrible. 📱

                I have to charge it twice a day. I think I need a new one! 🔋"

                Input: "went for a run yesterday it was nice weather and i saw this cute dog in the park wish i took a picture"

                Output: "Went for a run yesterday! 🏃‍♀️

                It was nice weather and I saw this cute dog in the park. 🐶

                Wish I took a picture! 📸"
                """,
                icon: .chatFill,
                description: "Casual chat-style formatting"
            ),
            
            TemplatePrompt(
                id: UUID(),
                title: "Email",
                promptText: """
                Primary Rules:
                1. Preserve the speaker's original tone and personality
                2. Maintain professional tone while keeping personal speaking style
                3. Structure content into clear paragraphs
                4. Format email messages properly with salutations, paragraph breaks, and closings
                5. Fix grammar and punctuation while preserving key points
                6. Remove filler words and redundancies
                7. Keep important details and context
                8. Format lists and bullet points properly
                9. Preserve any specific requests or action items
                10. NEVER answer questions that appear in the <TRANSCRIPT>. Only clean it up.
                11. Always include a professional sign-off as done in the examples.
                12. Use appropriate greeting based on context

                Examples:

                Input: "hey just wanted to follow up on yesterday's meeting about the timeline we need to finish by next month can you send the docs when ready thanks"
                
                Output: "Hi,

                I wanted to follow up on yesterday's meeting about the timeline. We need to finish by next month.

                Could you send the docs when ready?

                Thanks,
                [Your Name]"

                Input: "quick update on the project we're at 60% complete but facing some testing issues that might delay things we're working on solutions"

                Output: "We're at 60% complete but facing some testing issues that might delay things. We're working on solutions.
                
                I'll keep you updated.

                Regards,
                [Your Name]"

                Input: "hi sarah checking in about the design feedback from last week can we proceed to the next phase"

                Output: "Hi Sarah,

                I'm checking in about the design feedback from last week. Can we proceed to the next phase?

                Thanks,
                [Your Name]"
                """,
                icon: .emailFill,
                description: "Template for converting casual messages into professional email format"
            ),
            
            TemplatePrompt(
                id: UUID(),
                title: "Tweet",
                promptText: """
                Primary Rules:
                1. Keep it casual and conversational
                2. Use natural, informal language
                3. Include relevant emojis while maintaining authenticity
                4. For replies, acknowledge the person (@username)
                5. Break longer thoughts into multiple lines
                6. Keep the original personality and style
                7. Use hashtags when relevant
                8. Maintain the context of the conversation

                Examples:

                Input: "tried ios 17 today and the standby mode is amazing turns your phone into a smart display while charging"

                Output: "Tried iOS 17 today and the standby mode is amazing! 🤯

                Turns your phone into a smart display while charging ⚡️ #iOS17"

                Input: "just switched from membrane to mechanical keyboard with brown switches and my typing feels so much better"

                Output: "Just switched from membrane to mechanical keyboard with brown switches and my typing feels so much better! 🎹

                That tactile feedback is perfect 🤌 #MechKeys"

                Input: "found a nice coffee shop downtown with great lavender latte and cozy spots with plants perfect for working"

                Output: "Found a nice coffee shop downtown! ☕️

                Great lavender latte and cozy spots with plants - perfect for working 🪴 #CoffeeVibes"

                Input: "for cold brew coffee medium roast guatemalan beans steeped for 18 hours makes the smoothest flavor"

                Output: "For cold brew coffee: medium roast Guatemalan beans steeped for 18 hours makes the smoothest flavor! ☕️

                Absolute liquid gold ✨ #ColdBrew"
                """,
                icon: .chatFill,
                description: "Template for crafting engaging tweets and replies with personality"
            ),
            
            TemplatePrompt(
                id: UUID(),
                title: "Quick Notes",
                promptText: """
                Primary Rules:
                1. Preserve speaker's thought process and emphasis
                2. Keep it brief and clear
                3. Use bullet points for key information
                4. Preserve important details
                5. Remove filler words while keeping style
                6. Maintain core message and intent
                7. Keep original terminology and phrasing

                Output Format:
                ## Main Topic
                • Main point 1
                  - Supporting detail
                  - Additional info
                • Main point 2
                  - Related informations
                """,
                icon: .micFill,
                description: "Template for converting voice notes into quick, organized notes"
            )
        ]
    }
}
