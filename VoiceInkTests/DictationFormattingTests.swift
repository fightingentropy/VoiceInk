// Ported from FluidVoice (https://github.com/altic-dev/FluidVoice), GPL-3.0.
// Unit tests over SpokenPunctuationFormatter and DictationLiteralFormatter,
// mirroring FluidVoice's DictationE2ETests expectations.

import Testing
@testable import VoiceInk

struct DictationFormattingTests {

    // MARK: - Spoken punctuation

    @Test func spokenPunctuationConvertsCommonPhrases() {
        #expect(
            SpokenPunctuationFormatter.apply(
                "Hello comma world question mark open paren yes close paren quote done quote"
            ) == "Hello, world? (yes) \"done\""
        )
    }

    @Test func spokenPunctuationConvertsCodeAndContactPunctuation() {
        #expect(
            SpokenPunctuationFormatter.apply(
                "email at the rate example dot com slash help underscore me"
            ) == "email@example.com/help_me"
        )
        #expect(
            SpokenPunctuationFormatter.apply(
                "email at sign example dot com",
                frontmostAppContext: "com.openai.codex"
            ) == "email@example.com"
        )
        #expect(
            SpokenPunctuationFormatter.apply(
                "email at sign example dot com",
                frontmostAppContext: "Cursor com.todesktop.230313mzl4w4u92"
            ) == "email@example.com"
        )
        #expect(
            SpokenPunctuationFormatter.apply("email at sign example") == "email at sign example"
        )
        #expect(
            SpokenPunctuationFormatter.apply("x hyphen ray costs 50 percent") == "x-ray costs 50%"
        )
        #expect(
            SpokenPunctuationFormatter.apply("a plus b equals c") == "a + b = c"
        )
        #expect(
            SpokenPunctuationFormatter.apply("plus I need the normal word") == "plus I need the normal word"
        )
    }

    @Test func spokenPunctuationKeepsUngatedSingleWordsInProse() {
        #expect(
            SpokenPunctuationFormatter.apply("growing at the rate of 5 percent per year")
                == "growing at the rate of 5% per year"
        )
        #expect(
            SpokenPunctuationFormatter.apply("a million dollar idea") == "a million dollar idea"
        )
        #expect(
            SpokenPunctuationFormatter.apply("the door shut with a bang") == "the door shut with a bang"
        )
        #expect(
            SpokenPunctuationFormatter.apply("a large percent of users") == "a large percent of users"
        )
        #expect(
            SpokenPunctuationFormatter.apply("hash browns and a water pipe")
                == "hash browns and a water pipe"
        )
        #expect(
            SpokenPunctuationFormatter.apply("dollar 50 off") == "$50 off"
        )
    }

    @Test func spokenPunctuationGatesAtSignOnAppContext() {
        #expect(
            SpokenPunctuationFormatter.apply(
                "email at sign example dot com",
                frontmostAppContext: "com.apple.Notes"
            ) == "email at sign example.com"
        )
        #expect(
            SpokenPunctuationFormatter.apply(
                "email at sign example dot com",
                frontmostAppContext: nil
            ) == "email at sign example.com"
        )
    }

    @Test func spokenPunctuationKeepsBareDotInProse() {
        #expect(
            SpokenPunctuationFormatter.apply("the polka dot dress") == "the polka dot dress"
        )
        #expect(
            SpokenPunctuationFormatter.apply("example dot com") == "example.com"
        )
        #expect(
            SpokenPunctuationFormatter.apply("version 1 dot 2") == "version 1.2"
        )
    }

    @Test func spokenPunctuationPreservesSlashCommandSpacing() {
        let text = SpokenPunctuationFormatter.apply("Run slash status and open src slash services")
        #expect(text == "Run slash status and open src/services")
        #expect(
            DictationLiteralFormatter.applySlashCommandFormatting(text) == "Run /status and open src/services"
        )
    }

    @Test func spokenPunctuationCleansGeneratedCommaNoise() {
        #expect(
            SpokenPunctuationFormatter.apply("hyphen comma hyphen comma hyphen") == "---"
        )
        #expect(
            SpokenPunctuationFormatter.apply("50 comma percent") == "50%"
        )
        #expect(
            SpokenPunctuationFormatter.apply("open bracket comma close bracket") == "[]"
        )
        #expect(
            SpokenPunctuationFormatter.apply("open paren comma close paren") == "()"
        )
        #expect(
            SpokenPunctuationFormatter.apply("question mark comma exclamation mark") == "?!"
        )
    }

    @Test func spokenPunctuationPreservesExistingCommasNearSymbols() {
        #expect(
            SpokenPunctuationFormatter.apply("Thanks, @Sam") == "Thanks, @Sam"
        )
        #expect(
            SpokenPunctuationFormatter.apply("Use C++, now") == "Use C++, now"
        )
        #expect(
            SpokenPunctuationFormatter.apply("-,-,-") == "-,-,-"
        )
        #expect(
            SpokenPunctuationFormatter.apply("50, %") == "50, %"
        )
    }

    // MARK: - Slash commands

    @Test func slashCommandFormattingNormalizesSpokenAndLiteralCommands() {
        #expect(
            DictationLiteralFormatter.applySlashCommandFormatting("Run slash status and then / model.")
                == "Run /status and then /model."
        )
        #expect(
            DictationLiteralFormatter.applySlashCommandFormatting("Type forward slash fix-ci.")
                == "Type /fix-ci."
        )
        #expect(
            DictationLiteralFormatter.applySlashCommandFormatting("slash compact") == "/compact"
        )
    }

    @Test func slashCommandFormattingLeavesNonCommandSlashUsageAlone() {
        let text = "Use 1/2 and and/or. Open src slash services. Go to https slash slash example dot com. Slash and burn."

        #expect(DictationLiteralFormatter.applySlashCommandFormatting(text) == text)
    }

    // MARK: - Mentions

    @Test func mentionFormattingExplicitPhrasesWorkWithoutAppContext() {
        #expect(
            DictationLiteralFormatter.applyMentionFormatting("tag Paul") == "@Paul"
        )
        #expect(
            DictationLiteralFormatter.applyMentionFormatting("mention Paul Heinz, please") == "@Paul Heinz, please"
        )
        #expect(
            DictationLiteralFormatter.applyMentionFormatting("at sign maxgaav") == "@maxgaav"
        )
        #expect(
            DictationLiteralFormatter.applyMentionFormatting("at the rate Sarah") == "@Sarah"
        )
        #expect(
            DictationLiteralFormatter.applyMentionFormatting("mention Paul please") == "@Paul please"
        )
        #expect(
            DictationLiteralFormatter.applyMentionFormatting("tag Paul tomorrow") == "@Paul tomorrow"
        )
    }

    @Test func mentionFormattingRelaxedAtNameRequiresMentionAppContext() {
        #expect(
            DictationLiteralFormatter.applyMentionFormatting(
                "at Paul can you check this",
                frontmostAppContext: "com.tinyspeck.slackmacgap"
            ) == "@Paul can you check this"
        )
        #expect(
            DictationLiteralFormatter.applyMentionFormatting(
                "hey at Paul Heinz can you check this",
                frontmostAppContext: "com.hnc.Discord"
            ) == "hey @Paul Heinz can you check this"
        )
        #expect(
            DictationLiteralFormatter.applyMentionFormatting(
                "at Paul can you check this",
                frontmostAppContext: "com.apple.Notes"
            ) == "at Paul can you check this"
        )
    }

    @Test func mentionFormattingLeavesProseAlone() {
        let text = "I am at the store. Meet me at lunch. I am at Paul. Look at Paul's message."

        #expect(
            DictationLiteralFormatter.applyMentionFormatting(
                text,
                frontmostAppContext: "com.tinyspeck.slackmacgap"
            ) == text
        )
    }

    @Test func mentionFormattingLeavesTagAndMentionProseAlone() {
        for text in [
            "Did I mention Berlin is lovely",
            "The price tag was too high",
            "mention me in the thread",
            "tag along with us",
            "I will tag you later",
        ] {
            #expect(DictationLiteralFormatter.applyMentionFormatting(text) == text)
        }
    }

    // MARK: - Terminal autocomplete spacing

    @Test func terminalLiteralAutocompleteSpacingRemovesTrailingSpaceForTargetApps() {
        #expect(
            DictationLiteralFormatter.applyTerminalLiteralAutocompleteSpacing(
                "/model ",
                frontmostAppContext: "com.openai.codex"
            ) == "/model"
        )
        #expect(
            DictationLiteralFormatter.applyTerminalLiteralAutocompleteSpacing(
                "/model ",
                frontmostAppContext: "ChatGPT com.openai.chat"
            ) == "/model"
        )
        #expect(
            DictationLiteralFormatter.applyTerminalLiteralAutocompleteSpacing(
                "hey @Paul Heinz ",
                frontmostAppContext: "com.tinyspeck.slackmacgap"
            ) == "hey @Paul Heinz"
        )
        #expect(
            DictationLiteralFormatter.applyTerminalLiteralAutocompleteSpacing(
                " @Paul ",
                frontmostAppContext: "com.tinyspeck.slackmacgap"
            ) == " @Paul"
        )
        #expect(
            DictationLiteralFormatter.applyTerminalLiteralAutocompleteSpacing(
                "@ross.winn ",
                frontmostAppContext: "com.hnc.Discord"
            ) == "@ross.winn"
        )
    }

    @Test func terminalLiteralAutocompleteSpacingLeavesNonAutocompleteTextAlone() {
        #expect(
            DictationLiteralFormatter.applyTerminalLiteralAutocompleteSpacing(
                "/model ",
                frontmostAppContext: "com.apple.Notes"
            ) == "/model "
        )
        #expect(
            DictationLiteralFormatter.applyTerminalLiteralAutocompleteSpacing(
                "Run /status please ",
                frontmostAppContext: "com.openai.codex"
            ) == "Run /status please "
        )
        #expect(
            DictationLiteralFormatter.applyTerminalLiteralAutocompleteSpacing(
                "@Paul can you check this ",
                frontmostAppContext: "com.tinyspeck.slackmacgap"
            ) == "@Paul can you check this "
        )
    }

    // MARK: - Combined literal formatting

    @Test func combinedLiteralFormattingAppliesSlashCommandsThenMentions() {
        #expect(
            DictationLiteralFormatter.apply(
                "Run slash status and tag Paul",
                frontmostAppContext: "com.tinyspeck.slackmacgap"
            ) == "Run /status and @Paul"
        )
    }
}
